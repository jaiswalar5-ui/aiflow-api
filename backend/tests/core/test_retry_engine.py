import pytest
import asyncio
from unittest.mock import patch, MagicMock

from app.core.retry_engine import RetryEngine, RetryPolicy, RetryState
from app.providers.errors import (
    ProviderError, ErrorType, InvalidRequestError, AuthenticationError,
    RateLimitError, QuotaExhaustedError, TimeoutError, UnknownError, NetworkError
)


@pytest.fixture
def base_policy():
    return RetryPolicy(
        max_attempts=3,
        base_delay=0.1,
        max_delay=1.0,
        jitter_factor=0.0
    )


@pytest.mark.asyncio
async def test_successful_execution(base_policy):
    engine = RetryEngine(base_policy)
    
    async def mock_success():
        return "success"
        
    result = await engine.execute_with_retry(mock_success)
    assert result == "success"


@pytest.mark.asyncio
async def test_non_retryable_errors_abort_immediately(base_policy):
    engine = RetryEngine(base_policy)
    
    call_count = 0
    async def mock_fail_invalid():
        nonlocal call_count
        call_count += 1
        raise InvalidRequestError("Bad request")
        
    with pytest.raises(InvalidRequestError):
        await engine.execute_with_retry(mock_fail_invalid)
    assert call_count == 1  # 0 retries

    call_count = 0
    async def mock_fail_auth():
        nonlocal call_count
        call_count += 1
        raise AuthenticationError("Auth fail")
        
    with pytest.raises(AuthenticationError):
        await engine.execute_with_retry(mock_fail_auth)
    assert call_count == 1  # 0 retries


@pytest.mark.asyncio
async def test_retryable_error_reaches_max_attempts(base_policy):
    engine = RetryEngine(base_policy)
    
    call_count = 0
    async def mock_fail_timeout():
        nonlocal call_count
        call_count += 1
        raise TimeoutError("Timeout")
        
    with patch("asyncio.sleep") as mock_sleep:
        with pytest.raises(TimeoutError):
            await engine.execute_with_retry(mock_fail_timeout)
            
    # max_attempts is 3, so it should call the function 4 times (1 + 3 retries)? 
    # Wait, in code:
    # state.attempts is incremented before call
    # if state.attempts > max_attempts: raise
    # So max_attempts=3 means 3 total attempts (1 initial + 2 retries)? Or 1 initial + 3 retries?
    # checking code:
    # state.attempts += 1 (attempt 1) -> fail
    # attempt 1 > max_attempts (3)? No.
    # retry (sleep)
    # state.attempts += 1 (attempt 2) -> fail
    # attempt 2 > 3? No.
    # retry (sleep)
    # state.attempts += 1 (attempt 3) -> fail
    # attempt 3 > 3? No.
    # retry (sleep)
    # state.attempts += 1 (attempt 4) -> fail
    # attempt 4 > 3? Yes. raise.
    # So call_count is 4. (1 initial + 3 retries).
    assert call_count == 4
    assert mock_sleep.call_count == 3


@pytest.mark.asyncio
async def test_exponential_backoff_and_jitter():
    policy = RetryPolicy(max_attempts=3, base_delay=1.0, max_delay=10.0, jitter_factor=0.2)
    engine = RetryEngine(policy)
    
    # attempt 1 delay
    delay1 = engine._calculate_delay(1)
    # base = 1.0 * (2^0) = 1.0. Jitter = 20%
    assert 0.8 <= delay1 <= 1.2
    
    # attempt 2 delay
    delay2 = engine._calculate_delay(2)
    # base = 1.0 * (2^1) = 2.0. Jitter = 20%
    assert 1.6 <= delay2 <= 2.4


@pytest.mark.asyncio
async def test_retry_after_header_handling(base_policy):
    engine = RetryEngine(base_policy)
    
    call_count = 0
    async def mock_rate_limit():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RateLimitError("Rate limited", response_headers={"Retry-After": "5"})
        return "success"
        
    with patch("asyncio.sleep") as mock_sleep:
        result = await engine.execute_with_retry(mock_rate_limit)
        
    assert result == "success"
    assert call_count == 2
    mock_sleep.assert_called_once_with(5.0)


@pytest.mark.asyncio
async def test_unknown_error_conservative_policy():
    policy = RetryPolicy(max_attempts=5, conservative_unknown_retry=False)
    engine = RetryEngine(policy)
    
    call_count = 0
    async def mock_unknown():
        nonlocal call_count
        call_count += 1
        raise UnknownError("Unknown")
        
    with patch("asyncio.sleep") as mock_sleep:
        with pytest.raises(UnknownError):
            await engine.execute_with_retry(mock_unknown)
            
    # Should only retry once.
    # attempt 1 -> fail
    # conservatively stop checked at attempt 1 => True
    # raise immediately without sleeping
    assert call_count == 1
    assert mock_sleep.call_count == 0

    # With conservative_unknown_retry = True, it should retry normally
    policy.conservative_unknown_retry = True
    engine2 = RetryEngine(policy)
    
    call_count = 0
    with patch("asyncio.sleep") as mock_sleep:
        with pytest.raises(UnknownError):
            await engine2.execute_with_retry(mock_unknown)
            
    assert call_count == 6  # 1 initial + 5 retries


@pytest.mark.asyncio
async def test_network_and_timeout_scenarios(base_policy):
    engine = RetryEngine(base_policy)
    
    # Network Error
    call_count = 0
    async def mock_network():
        nonlocal call_count
        call_count += 1
        raise NetworkError("Network fail")
        
    with patch("asyncio.sleep"):
        with pytest.raises(NetworkError):
            await engine.execute_with_retry(mock_network)
            
    assert call_count == 4


def test_log_sanitization():
    from app.providers.errors import ProviderError
    
    e = InvalidRequestError("Invalid key sk-somekey12345678901234567890")
    assert "sk-somekey" not in e.safe_message
    
    # Validate the engine logs don't include raw error messages that bypass sanitization
    # Since engine logs e.error_type it naturally avoids it, and request_id is just uuid.
    # This just ensures we don't regress.
