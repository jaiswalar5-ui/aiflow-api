from typing import Any, Callable, Coroutine, TypeVar, Optional, Dict
from pydantic import BaseModel, Field
import asyncio
import random
import time
import uuid

from app.providers.errors import (
    ProviderError,
    ErrorType,
    RateLimitError,
    QuotaExhaustedError
)
from app.core.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")

class RetryPolicy(BaseModel):
    """Configuration for retry execution."""
    max_attempts: int = Field(default=3, description="Maximum number of retry attempts (0 means no retries)")
    base_delay: float = Field(default=1.0, description="Base delay in seconds for exponential backoff")
    max_delay: float = Field(default=30.0, description="Maximum delay in seconds between retries")
    jitter_factor: float = Field(default=0.2, description="Random jitter factor to add to delays")
    conservative_unknown_retry: bool = Field(default=False, description="Whether to aggressively retry unknown errors (False means max 1 retry for unknown)")

class RetryState:
    """State of the current retry execution."""
    def __init__(self, policy: RetryPolicy, request_id: Optional[str] = None):
        self.policy = policy
        self.attempts = 0
        self.request_id = request_id or str(uuid.uuid4())
        self.aborted = False

class RetryEngine:
    """Engine executing async callbacks with retry and backoff logic."""
    def __init__(self, policy: Optional[RetryPolicy] = None):
        self.policy = policy or RetryPolicy()

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and jitter."""
        base = self.policy.base_delay * (2 ** (attempt - 1))
        delay = min(base, self.policy.max_delay)
        
        if self.policy.jitter_factor > 0:
            jitter = delay * self.policy.jitter_factor
            delay = delay + random.uniform(-jitter, jitter)
            
        return max(0.0, delay)
    
    async def execute_with_retry(
        self,
        func: Callable[..., Coroutine[Any, Any, T]],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Execute an async function with the configured retry policy."""
        state = RetryState(self.policy, kwargs.get('request_id'))
        
        while True:
            try:
                state.attempts += 1
                return await func(*args, **kwargs)
            except ProviderError as e:
                # Basic non-retryable check
                if not e.retryable:
                    logger.debug(f"Operation failed with non-retryable error: {e.error_type} [Request {state.request_id}]")
                    raise
                
                # Check bounds
                if state.attempts > self.policy.max_attempts:
                    logger.warning(
                        f"Max retries ({self.policy.max_attempts}) reached. "
                        f"Failing with {e.error_type} [Request {state.request_id}]"
                    )
                    raise
                
                # Calculate basic backoff
                delay = self._calculate_delay(state.attempts)
                
                logger.info(
                    f"Retry {state.attempts}/{self.policy.max_attempts} for "
                    f"request {state.request_id} after {delay:.2f}s delay. "
                    f"Reason: {e.error_type}"
                )
                
                await asyncio.sleep(delay)

