"""
Multi-Provider Failover Engine (Task 13)

Centralises all failover orchestration logic. The engine:
  - Iterates an ordered list of eligible providers (cloud first, local-fallback last).
  - Delegates per-provider retry/backoff to the Task 12 RetryEngine.
  - Uses Task 11 ErrorClassifier results (via ProviderError.error_type / .retryable)
    to decide whether to fail-over or abort immediately.  No provider error mapping
    is duplicated here.
  - Preserves a single correlation/request ID across the entire failover chain.
  - Records provider attempt number and failover reason in structured logs.
  - Prevents duplicate responses and request loops via a visited-provider set and
    a configurable max_provider_attempts ceiling.
  - Falls back to local providers (is_local_fallback=True) only after all cloud
    providers have been tried.
  - Raises AllProvidersExhaustedError (503) when every provider including local
    has failed; the error contains a safe message and machine-readable error code.
  - Never exposes individual provider credentials or internal infrastructure details.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

from app.core.logging import get_logger
from app.core.retry_engine import RetryEngine, RetryPolicy
from app.providers.errors import (
    AllProvidersExhaustedError,
    AuthenticationError,
    ErrorType,
    InvalidRequestError,
    ProviderError,
    UnsupportedModelError,
)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Failover policy
# ---------------------------------------------------------------------------


class FailoverPolicy(BaseModel):
    """Configuration knobs for the failover engine.

    All decisions about *when* to failover live here so that the engine
    itself contains no hard-coded provider logic.
    """

    max_provider_attempts: int = Field(
        default=5,
        ge=1,
        le=20,
        description=(
            "Absolute ceiling on the number of distinct providers attempted "
            "across the full failover chain.  Prevents infinite loops when a "
            "large number of providers are configured."
        ),
    )
    failover_on_auth_error: bool = Field(
        default=False,
        description=(
            "When False (default) an authentication / configuration failure "
            "stops the chain immediately — credentials are wrong for ALL "
            "providers equally and trying the next one wastes time."
        ),
    )
    failover_on_invalid_request: bool = Field(
        default=False,
        description=(
            "When False (default) a malformed / semantically invalid request "
            "is not retried across providers — the request itself is broken."
        ),
    )
    failover_on_unsupported_model: bool = Field(
        default=True,
        description=(
            "When True (default) an unsupported-model error causes failover to "
            "the next provider, but ONLY when that provider advertises support "
            "for the requested model (guarded failover)."
        ),
    )


# ---------------------------------------------------------------------------
# Per-attempt record (structured logging)
# ---------------------------------------------------------------------------


@dataclass
class ProviderAttemptRecord:
    """Structured record of a single provider attempt within a failover chain."""

    provider_name: str
    attempt_number: int          # 1-based index across the chain
    error_type: Optional[str] = None
    failover_reason: Optional[str] = None
    succeeded: bool = False

    def as_log_dict(self, correlation_id: str) -> Dict[str, Any]:
        return {
            "correlation_id": correlation_id,
            "provider": self.provider_name,
            "attempt_number": self.attempt_number,
            "error_type": self.error_type,
            "failover_reason": self.failover_reason,
            "succeeded": self.succeeded,
        }


# ---------------------------------------------------------------------------
# Failover engine
# ---------------------------------------------------------------------------


class FailoverEngine:
    """Orchestrates multi-provider failover for a single gateway request.

    All dependencies are injected; no singleton imports inside methods.
    """

    def __init__(
        self,
        retry_engine: RetryEngine,
        failover_policy: Optional[FailoverPolicy] = None,
    ) -> None:
        self._retry_engine = retry_engine
        self._policy = failover_policy or FailoverPolicy()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def execute_with_failover(
        self,
        func_factory: Callable[[str], Callable[[], Coroutine[Any, Any, Any]]],
        cloud_providers: List[str],
        local_providers: List[str],
        correlation_id: Optional[str] = None,
        model: Optional[str] = None,
        provider_models: Optional[Dict[str, List[str]]] = None,
    ) -> Any:
        """Attempt the request across an ordered sequence of providers.

        Args:
            func_factory:     Given a *provider_name* returns a **no-arg async
                              callable** (not a coroutine) so that the retry engine
                              can call it multiple times.
                              Example::

                                  lambda name: lambda: registry.get_provider(name) \\
                                      .send_chat_completion(request)

            cloud_providers:  Ordered list of cloud provider names (priority-sorted).
            local_providers:  Ordered list of local/fallback provider names tried
                              only after every cloud provider has failed.
            correlation_id:   Single request-scoped ID preserved across the chain.
                              Auto-generated when not supplied.
            model:            The model name requested (for unsupported-model guard).
            provider_models:  Mapping ``{provider_name: [supported_model, ...]}``
                              used for guarded unsupported-model failover.

        Returns:
            The first successful response.

        Raises:
            ProviderError:             Immediately on auth / invalid-request errors
                                       when failover is disabled by policy.
            AllProvidersExhaustedError: When every provider has failed.
        """
        cid = correlation_id or str(uuid.uuid4())
        provider_models = provider_models or {}

        # Build the full ordered sequence: cloud first, then local.
        ordered: List[Tuple[str, bool]] = [  # (name, is_local)
            *((p, False) for p in cloud_providers),
            *((p, True) for p in local_providers),
        ]

        if not ordered:
            raise AllProvidersExhaustedError(
                message="No providers are configured.",
                request_id=cid,
            )

        visited: Set[str] = set()
        attempt_records: List[ProviderAttemptRecord] = []
        last_error: Optional[ProviderError] = None
        attempt_number = 0

        for provider_name, is_local in ordered:
            # ── Loop-prevention guard ───────────────────────────────────────
            if provider_name in visited:
                logger.warning(
                    "Skipping already-visited provider to prevent loop.",
                    extra={
                        "correlation_id": cid,
                        "provider": provider_name,
                    },
                )
                continue

            if attempt_number >= self._policy.max_provider_attempts:
                logger.warning(
                    "max_provider_attempts reached; aborting failover chain.",
                    extra={
                        "correlation_id": cid,
                        "max_provider_attempts": self._policy.max_provider_attempts,
                        "attempt_number": attempt_number,
                    },
                )
                break

            visited.add(provider_name)
            attempt_number += 1
            record = ProviderAttemptRecord(
                provider_name=provider_name,
                attempt_number=attempt_number,
            )
            attempt_records.append(record)

            logger.info(
                "Attempting provider.",
                extra={
                    "correlation_id": cid,
                    "provider": provider_name,
                    "attempt_number": attempt_number,
                    "is_local_fallback": is_local,
                },
            )

            # ── Per-provider attempt (with retry/backoff from Task 12) ──────
            try:
                provider_func = func_factory(provider_name)
                result = await self._retry_engine.execute_with_retry(provider_func)
                record.succeeded = True
                logger.info(
                    "Provider succeeded.",
                    extra=record.as_log_dict(cid),
                )
                return result

            except ProviderError as exc:
                record.error_type = exc.error_type.value
                last_error = exc

                # ── Policy: should we failover? ──────────────────────────
                should_failover, reason = self._evaluate_failover(
                    exc, provider_name, model, provider_models, ordered, visited
                )
                record.failover_reason = reason

                logger.warning(
                    "Provider failed.",
                    extra={
                        **record.as_log_dict(cid),
                        "failover_decision": should_failover,
                    },
                )

                if not should_failover:
                    # Abort the whole chain — re-raise the original error.
                    raise

                # Continue to next provider in the loop.
                continue

        # ── All providers exhausted ─────────────────────────────────────────
        logger.error(
            "All providers exhausted.",
            extra={
                "correlation_id": cid,
                "providers_tried": [r.provider_name for r in attempt_records],
                "total_attempts": attempt_number,
            },
        )
        raise AllProvidersExhaustedError(
            message=(
                "The request could not be completed. All configured providers "
                "are currently unavailable. Please try again later."
            ),
            request_id=cid,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evaluate_failover(
        self,
        exc: ProviderError,
        current_provider: str,
        model: Optional[str],
        provider_models: Dict[str, List[str]],
        ordered: List[Tuple[str, bool]],
        visited: Set[str],
    ) -> Tuple[bool, str]:
        """Return (should_failover, human-readable reason).

        Policy decisions (no provider-specific logic):
        1. Non-retryable errors abort by default, with explicit policy exceptions.
        2. Auth errors: respect ``failover_on_auth_error``.
        3. Invalid request: respect ``failover_on_invalid_request``.
        4. Unsupported model: only failover when a remaining provider supports it.
        5. All other retryable (server/network/timeout/rate-limit) → failover.
        """
        et = exc.error_type

        # Auth failures
        if et == ErrorType.AUTHENTICATION_ERROR:
            if self._policy.failover_on_auth_error:
                return True, "authentication_error_with_failover_enabled"
            return False, "authentication_error_no_failover"

        # Malformed / semantically invalid requests
        if et == ErrorType.INVALID_REQUEST:
            if self._policy.failover_on_invalid_request:
                return True, "invalid_request_with_failover_enabled"
            return False, "invalid_request_no_failover"

        # Unsupported model — guarded failover
        if et == ErrorType.UNSUPPORTED_MODEL_ERROR:
            if not self._policy.failover_on_unsupported_model:
                return False, "unsupported_model_failover_disabled"
            if model:
                # Only failover if a remaining (unvisited) provider supports the model.
                remaining_support = any(
                    (p not in visited) and (model in provider_models.get(p, []))
                    for p, _ in ordered
                )
                if not remaining_support:
                    return (
                        False,
                        f"unsupported_model_no_capable_provider_remaining_for_{model}",
                    )
            return True, "unsupported_model_failover_to_capable_provider"

        # Any other retryable error → failover
        if exc.retryable:
            return True, f"retryable_{et.value}"

        # Non-retryable unknown — do NOT failover blindly.
        return False, f"non_retryable_{et.value}"


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------

_failover_engine: Optional[FailoverEngine] = None


def get_failover_engine(
    retry_engine: Optional[RetryEngine] = None,
    failover_policy: Optional[FailoverPolicy] = None,
) -> "FailoverEngine":
    """Return the module-level singleton FailoverEngine.

    Accepts optional overrides so tests can inject different configurations
    without monkey-patching globals.
    """
    global _failover_engine
    if _failover_engine is None or retry_engine is not None or failover_policy is not None:
        from app.core.retry_engine import RetryEngine as _RE, RetryPolicy as _RP

        _engine = retry_engine or _RE(policy=_RP())
        _failover_engine = FailoverEngine(
            retry_engine=_engine,
            failover_policy=failover_policy,
        )
    return _failover_engine


def set_failover_engine(engine: FailoverEngine) -> None:
    """Replace the global singleton (for testing)."""
    global _failover_engine
    _failover_engine = engine
