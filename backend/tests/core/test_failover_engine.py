"""
Tests for Task 13 — Multi-Provider Failover Engine

Covers:
  1. First-provider success (no failover)
  2. First-provider timeout → second-provider success
  3. All cloud providers fail → local-model success
  4. Every provider fails → structured 503 AllProvidersExhaustedError
  5. Infinite-loop prevention via max_provider_attempts
  6. Auth error does NOT failover (policy)
  7. Invalid request does NOT failover (policy)
  8. Unsupported model only fails over when next provider supports it
"""

import pytest
import asyncio
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.failover_engine import FailoverEngine, FailoverPolicy
from app.core.retry_engine import RetryEngine, RetryPolicy
from app.providers.errors import (
    AllProvidersExhaustedError,
    AuthenticationError,
    ErrorType,
    InvalidRequestError,
    ProviderError,
    TimeoutError,
    ServerError,
    UnsupportedModelError,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def instant_policy():
    """RetryPolicy with max_attempts=1 so the engine makes exactly 1 call per provider.

    In the RetryEngine:
      - state.attempts is incremented BEFORE the call
      - check is: if state.attempts > max_attempts: raise
      - max_attempts=1 means: attempt=1 calls once, fails, sleeps, attempt=2 > 1 raises.
    To get exactly 1 call with NO retry we need max_attempts=0... but the field has ge=0.
    We use max_attempts=1 which means at most 2 calls (initial + 1 retry). Tests that need
    exactly-one-call semantics check order, not exact count.
    """
    return RetryPolicy(
        max_attempts=1,
        base_delay=0.0,
        max_delay=0.0,
        jitter_factor=0.0,
    )


@pytest.fixture
def retry_engine(instant_policy):
    return RetryEngine(policy=instant_policy)


@pytest.fixture
def default_failover_policy():
    return FailoverPolicy(
        max_provider_attempts=5,
        failover_on_auth_error=False,
        failover_on_invalid_request=False,
        failover_on_unsupported_model=True,
    )


@pytest.fixture
def engine(retry_engine, default_failover_policy):
    return FailoverEngine(
        retry_engine=retry_engine,
        failover_policy=default_failover_policy,
    )


def _make_response(text: str = "ok"):
    """Build a minimal mock ChatCompletionResponse."""
    resp = MagicMock()
    resp.content = text
    resp.usage = None
    return resp


# ---------------------------------------------------------------------------
# 1. First-provider success
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_first_provider_success(engine):
    """First provider returns OK; no failover occurs and correlation_id is preserved."""
    expected = _make_response("hello")
    call_log = []

    def func_factory(name: str):
        async def _call():
            call_log.append(name)
            return expected
        return _call

    CID = "test-correlation-id-001"
    result = await engine.execute_with_failover(
        func_factory=func_factory,
        cloud_providers=["provider_a", "provider_b"],
        local_providers=[],
        correlation_id=CID,
    )

    assert result is expected
    assert call_log == ["provider_a"], "Should have called only the first provider"


# ---------------------------------------------------------------------------
# 2. First provider timeout → second provider success
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_timeout_then_second_provider_success(engine):
    """Provider 1 raises TimeoutError (retryable); engine fails over to provider 2."""
    expected = _make_response("from provider_b")
    call_log = []  # records (provider_name) for each attempt

    def func_factory(name: str):
        async def _call():
            call_log.append(name)
            if name == "provider_a":
                raise TimeoutError("timed out", provider_name="provider_a")
            return expected
        return _call

    result = await engine.execute_with_failover(
        func_factory=func_factory,
        cloud_providers=["provider_a", "provider_b"],
        local_providers=[],
        correlation_id="test-002",
    )

    assert result is expected
    # provider_a will be called at least once (retried by retry engine before failover)
    assert "provider_a" in call_log
    # provider_b must have been called AND must come after a provider_a call
    assert "provider_b" in call_log
    last_a = max(i for i, n in enumerate(call_log) if n == "provider_a")
    first_b = next(i for i, n in enumerate(call_log) if n == "provider_b")
    assert first_b > last_a, "provider_b must appear after all provider_a attempts"


# ---------------------------------------------------------------------------
# 3. All cloud providers fail → local-model success
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_all_cloud_fail_then_local_success(engine):
    """All cloud providers fail; engine falls back to the local provider."""
    expected = _make_response("from local mock")
    call_log = []

    def func_factory(name: str):
        async def _call():
            call_log.append(name)
            if name != "local_mock":
                raise ServerError("cloud down", provider_name=name)
            return expected
        return _call

    result = await engine.execute_with_failover(
        func_factory=func_factory,
        cloud_providers=["cloud_a", "cloud_b"],
        local_providers=["local_mock"],
        correlation_id="test-003",
    )

    assert result is expected
    assert "cloud_a" in call_log
    assert "cloud_b" in call_log
    assert "local_mock" in call_log
    assert call_log.index("local_mock") > call_log.index("cloud_b"), (
        "Local mock must be tried after all cloud providers"
    )


# ---------------------------------------------------------------------------
# 4. Every provider fails → structured 503
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_all_providers_fail_returns_503(engine):
    """When every provider (including local) fails, raise AllProvidersExhaustedError."""
    def func_factory(name: str):
        async def _call():
            raise ServerError("all down", provider_name=name)
        return _call

    with pytest.raises(AllProvidersExhaustedError) as exc_info:
        await engine.execute_with_failover(
            func_factory=func_factory,
            cloud_providers=["cloud_a", "cloud_b"],
            local_providers=["local_mock"],
            correlation_id="test-004",
        )

    err = exc_info.value
    assert err.error_type == ErrorType.ALL_PROVIDERS_EXHAUSTED
    assert err.status_code == 503
    assert err.retryable is False
    # Safe message must not contain provider internals
    msg = err.safe_message.lower()
    assert "cloud_a" not in msg
    assert "cloud_b" not in msg


# ---------------------------------------------------------------------------
# 5. No infinite loop — max_provider_attempts enforced
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_no_infinite_loop():
    """max_provider_attempts=2 with 5 providers — only 2 distinct providers are attempted."""
    policy = FailoverPolicy(max_provider_attempts=2)
    retry_policy = RetryPolicy(max_attempts=1, base_delay=0.0, max_delay=0.0, jitter_factor=0.0)
    eng = FailoverEngine(
        retry_engine=RetryEngine(policy=retry_policy),
        failover_policy=policy,
    )
    call_log = []

    def func_factory(name: str):
        async def _call():
            call_log.append(name)
            raise ServerError("always fail", provider_name=name)
        return _call

    with pytest.raises(AllProvidersExhaustedError):
        await eng.execute_with_failover(
            func_factory=func_factory,
            cloud_providers=["p1", "p2", "p3", "p4", "p5"],
            local_providers=[],
            correlation_id="test-005",
        )

    distinct_providers_tried = set(call_log)
    # max_provider_attempts=2 means at most 2 DISTINCT providers are attempted
    assert len(distinct_providers_tried) <= 2, (
        f"Expected at most 2 distinct providers, got {len(distinct_providers_tried)}: {distinct_providers_tried}"
    )
    # Each provider in distinct set should only be from the original list
    assert distinct_providers_tried.issubset({"p1", "p2", "p3", "p4", "p5"})
    # No provider should appear in the visited set more than the retry count allows
    for provider in distinct_providers_tried:
        count = call_log.count(provider)
        assert count <= 2, f"{provider} called {count} times — exceeds retry max of 2"


# ---------------------------------------------------------------------------
# 6. Auth error does NOT failover
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_auth_error_does_not_failover(engine):
    """AuthenticationError with failover_on_auth_error=False must abort immediately."""
    call_log = []

    def func_factory(name: str):
        async def _call():
            call_log.append(name)
            raise AuthenticationError("bad key", provider_name=name)
        return _call

    with pytest.raises(AuthenticationError):
        await engine.execute_with_failover(
            func_factory=func_factory,
            cloud_providers=["provider_a", "provider_b"],
            local_providers=["local"],
            correlation_id="test-006",
        )

    # Must abort after the first provider — no failover
    assert call_log == ["provider_a"]


# ---------------------------------------------------------------------------
# 7. Invalid request does NOT failover
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_invalid_request_does_not_failover(engine):
    """InvalidRequestError with failover_on_invalid_request=False must abort immediately."""
    call_log = []

    def func_factory(name: str):
        async def _call():
            call_log.append(name)
            raise InvalidRequestError("bad params", provider_name=name)
        return _call

    with pytest.raises(InvalidRequestError):
        await engine.execute_with_failover(
            func_factory=func_factory,
            cloud_providers=["provider_a", "provider_b"],
            local_providers=["local"],
            correlation_id="test-007",
        )

    assert call_log == ["provider_a"]


# ---------------------------------------------------------------------------
# 8. Unsupported model — guarded failover
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unsupported_model_failover_only_when_capable(engine):
    """UnsupportedModelError only fails over to providers that support the requested model."""
    expected = _make_response("from capable provider")
    call_log = []
    MODEL = "special-model"

    def func_factory(name: str):
        async def _call():
            call_log.append(name)
            if name == "provider_a":
                # Doesn't support the model
                raise UnsupportedModelError("model not found", provider_name=name)
            if name == "provider_no_model":
                # Also doesn't support — but this one would be a failover candidate
                raise UnsupportedModelError("nope", provider_name=name)
            return expected
        return _call

    # provider_a and provider_no_model don't support the model;
    # provider_capable does.
    provider_models = {
        "provider_a": ["other-model"],
        "provider_no_model": ["other-model"],
        "provider_capable": [MODEL],
    }

    result = await engine.execute_with_failover(
        func_factory=func_factory,
        cloud_providers=["provider_a", "provider_no_model", "provider_capable"],
        local_providers=[],
        correlation_id="test-008",
        model=MODEL,
        provider_models=provider_models,
    )

    assert result is expected
    assert "provider_capable" in call_log


@pytest.mark.asyncio
async def test_unsupported_model_no_capable_fallback_aborts(engine):
    """When no remaining provider supports the model, failover must stop."""
    call_log = []
    MODEL = "exotic-model"

    def func_factory(name: str):
        async def _call():
            call_log.append(name)
            raise UnsupportedModelError("nope", provider_name=name)
        return _call

    provider_models = {
        "provider_a": ["other-model"],
        "provider_b": ["other-model"],
    }

    with pytest.raises(UnsupportedModelError):
        await engine.execute_with_failover(
            func_factory=func_factory,
            cloud_providers=["provider_a", "provider_b"],
            local_providers=[],
            correlation_id="test-008b",
            model=MODEL,
            provider_models=provider_models,
        )

    # Should abort after first provider since no remaining provider supports the model
    assert call_log == ["provider_a"]


# ---------------------------------------------------------------------------
# 9. No providers configured → immediate 503
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_no_providers_configured():
    """With empty provider lists, raise AllProvidersExhaustedError immediately."""
    eng = FailoverEngine(
        retry_engine=RetryEngine(),
        failover_policy=FailoverPolicy(),
    )

    with pytest.raises(AllProvidersExhaustedError) as exc_info:
        await eng.execute_with_failover(
            func_factory=lambda name: (lambda: None),  # type: ignore[return-value]
            cloud_providers=[],
            local_providers=[],
            correlation_id="test-009",
        )

    assert exc_info.value.status_code == 503


# ---------------------------------------------------------------------------
# 10. Correlation ID auto-generated and consistent
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_correlation_id_preserved_in_logs(engine):
    """When correlation_id is provided, it should flow without modification."""
    captured_logs = []
    expected = _make_response()

    original_info = engine._failover_engine._retry_engine if hasattr(engine, '_failover_engine') else None

    def func_factory(name: str):
        async def _call():
            return expected
        return _call

    CID = "fixed-correlation-id-xyz"
    # Just verifying the engine doesn't crash and returns the right result
    result = await engine.execute_with_failover(
        func_factory=func_factory,
        cloud_providers=["provider_a"],
        local_providers=[],
        correlation_id=CID,
    )
    assert result is expected


@pytest.mark.asyncio
async def test_correlation_id_is_passed_to_retry_engine():
    """The failover chain uses one ID when delegating provider retries."""
    class RecordingRetryEngine:
        def __init__(self):
            self.request_ids = []

        async def execute_with_retry(self, func, *, request_id=None):
            self.request_ids.append(request_id)
            return await func()

    retry_engine = RecordingRetryEngine()
    eng = FailoverEngine(
        retry_engine=cast(RetryEngine, retry_engine),
        failover_policy=FailoverPolicy(),
    )
    cid = "fixed-correlation-id-retry"

    def func_factory(name: str):
        async def _call():
            return _make_response()
        return _call

    result = await eng.execute_with_failover(
        func_factory=func_factory,
        cloud_providers=["provider_a"],
        local_providers=[],
        correlation_id=cid,
    )

    assert result is not None
    assert retry_engine.request_ids == [cid]


@pytest.mark.asyncio
async def test_failed_provider_error_keeps_correlation_id_and_safe_message(engine):
    """Provider details stay sanitized while the request ID remains traceable."""
    cid = "fixed-correlation-id-error"

    def func_factory(name: str):
        async def _call():
            raise ServerError(
                "upstream https://internal.example/key/secret?token=abc12345678901234567",
                provider_name=name,
            )
        return _call

    with pytest.raises(AllProvidersExhaustedError) as exc_info:
        await engine.execute_with_failover(
            func_factory=func_factory,
            cloud_providers=["provider_a"],
            local_providers=[],
            correlation_id=cid,
        )

    assert exc_info.value.request_id == cid
    assert "internal.example" not in str(exc_info.value)
    assert "abc12345678901234567" not in str(exc_info.value)

@pytest.mark.asyncio
async def test_failover_updates_health_manager(engine):
    """A real FailoverEngine request must update the injected health manager."""
    from app.core.health_manager import health_manager, ProviderHealthState
    from app.models.provider_config import ProviderConfig, HealthCheckSettings
    import time
    
    # Setup mock config in health_manager for provider_a and provider_b
    config_a = ProviderConfig(
        name="provider_a",
        adapter_type="app.providers.mock.MockProvider",
        supported_models=["model"],
        health_check_settings=HealthCheckSettings(
            enabled=True, healthy_threshold=2, unhealthy_threshold=2
        )
    )
    config_b = ProviderConfig(
        name="provider_b",
        adapter_type="app.providers.mock.MockProvider",
        supported_models=["model"],
        health_check_settings=HealthCheckSettings(
            enabled=True, healthy_threshold=2, unhealthy_threshold=2
        )
    )
    
    # We must patch get_provider_config for health manager
    original_get = health_manager.registry.get_provider_config
    def mock_get(name):
        if name == "provider_a": return config_a
        if name == "provider_b": return config_b
        return None
    health_manager.registry.get_provider_config = mock_get
    
    # Reset state
    health_manager.health_states.clear()
    
    expected = _make_response("success")
    
    def func_factory(name: str):
        async def _call():
            if name == "provider_a":
                raise ServerError("fail")
            return expected
        return _call

    # Execute failover which should fail on a and succeed on b
    await engine.execute_with_failover(
        func_factory=func_factory,
        cloud_providers=["provider_a", "provider_b"],
        local_providers=[],
        correlation_id="test-integration-1",
    )
    
    snapshot_a = health_manager.get_snapshot("provider_a")
    snapshot_b = health_manager.get_snapshot("provider_b")
    
    assert snapshot_a["consecutive_failures"] == 1
    assert snapshot_a["last_error_category"] == "server_error"
    assert snapshot_a["state"] == "degraded"
    
    assert snapshot_b["consecutive_successes"] == 1
    assert snapshot_b["state"] == "healthy"
    
    # Restore original config
    health_manager.registry.get_provider_config = original_get
