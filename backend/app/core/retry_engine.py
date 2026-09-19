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

