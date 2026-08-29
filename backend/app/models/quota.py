from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class ProviderAvailability(str, Enum):
    """Provider availability state."""
    AVAILABLE = "available"
    COOLDOWN = "cooldown"
    EXHAUSTED = "exhausted"


class QuotaConfidence(str, Enum):
    """Confidence level in quota information."""
    CONFIRMED = "confirmed"  # Provider explicitly exposes quota info
    ESTIMATED = "estimated"  # Derived from rate limit responses
    UNKNOWN = "unknown"  # No quota information available


class QuotaResetInfo(BaseModel):
    """Information about quota reset timing if exposed by provider."""
    reset_time: Optional[datetime] = Field(default=None, description="When quota resets")
    reset_window_seconds: Optional[int] = Field(default=None, description="Reset window in seconds")
    requests_remaining: Optional[int] = Field(default=None, description="Requests remaining in current window")
    tokens_remaining: Optional[int] = Field(default=None, description="Tokens remaining in current window")


class ProviderQuotaState(BaseModel):
    """Comprehensive quota and rate limit state for a provider."""
    provider_name: str = Field(..., description="Provider identifier")
    availability: ProviderAvailability = Field(default=ProviderAvailability.AVAILABLE, description="Current availability state")
    quota_confidence: QuotaConfidence = Field(default=QuotaConfidence.UNKNOWN, description="Confidence in quota info")
    
    # Usage tracking
    request_count: int = Field(default=0, description="Total requests made")
    token_usage: int = Field(default=0, description="Total tokens consumed")
    rate_limit_events: int = Field(default=0, description="Number of rate limit events encountered")
    
    # Timing and state management
    last_quota_failure: Optional[datetime] = Field(default=None, description="Last time quota was exceeded")
    cooldown_until: Optional[datetime] = Field(default=None, description="When cooldown period ends")
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Last state update")
    
    # Provider-specific quota info (if available)
    reset_info: Optional[QuotaResetInfo] = Field(default=None, description="Quota reset information from provider")
    
    # Error tracking for cooldown decisions
    consecutive_failures: int = Field(default=0, description="Consecutive rate limit failures")
    last_429_response: Optional[Dict[str, Any]] = Field(default=None, description="Last 429 response details")
    
    def is_available(self) -> bool:
        """Check if provider is currently available for requests."""
        if self.availability == ProviderAvailability.EXHAUSTED:
            return False
        
        if self.availability == ProviderAvailability.COOLDOWN:
            if self.cooldown_until and datetime.now(timezone.utc) < self.cooldown_until:
                return False
            # Cooldown expired, mark as available
            self.availability = ProviderAvailability.AVAILABLE
            self.cooldown_until = None
            self.consecutive_failures = 0
            
        return True
    
    def record_request(self, tokens_used: int = 0) -> None:
        """Record a successful request."""
        self.request_count += 1
        self.token_usage += tokens_used
        self.last_updated = datetime.now(timezone.utc)
        
        # Reset consecutive failures on success
        if self.consecutive_failures > 0:
            self.consecutive_failures = 0
            if self.availability == ProviderAvailability.COOLDOWN:
                self.availability = ProviderAvailability.AVAILABLE
                self.cooldown_until = None
    
    def record_rate_limit(self, response_details: Optional[Dict[str, Any]] = None, cooldown_seconds: int = 60) -> None:
        """Record a rate limit event and update availability state."""
        self.rate_limit_events += 1
        self.consecutive_failures += 1
        self.last_quota_failure = datetime.now(timezone.utc)
        self.last_429_response = response_details
        self.last_updated = datetime.now(timezone.utc)
        
        # Determine availability state based on consecutive failures
        if self.consecutive_failures >= 3:
            # Multiple consecutive failures - mark as exhausted
            self.availability = ProviderAvailability.EXHAUSTED
        else:
            # Single or few failures - put in cooldown
            self.availability = ProviderAvailability.COOLDOWN
            self.cooldown_until = (datetime.now(timezone.utc).replace(
                microsecond=0
            ) + timedelta(seconds=cooldown_seconds))
    
    def update_reset_info(self, reset_info: QuotaResetInfo) -> None:
        """Update quota reset information from provider response."""
        self.reset_info = reset_info
        self.quota_confidence = QuotaConfidence.CONFIRMED
        self.last_updated = datetime.now(timezone.utc)
    
    def get_estimated_quota_health(self) -> float:
        """
        Get estimated quota health score (0.0 to 1.0).
        Higher is better. Based on rate limit events and recent failures.
        """
        if self.availability == ProviderAvailability.EXHAUSTED:
            return 0.0
        
        if self.availability == ProviderAvailability.COOLDOWN:
            # Return partial score based on cooldown progress
            if self.cooldown_until:
                remaining = (self.cooldown_until - datetime.now(timezone.utc)).total_seconds()
                if remaining > 0:
                    return max(0.0, 1.0 - (remaining / 3600.0))  # Decay over 1 hour
            return 0.5
        
        # Base score decayed by rate limit events
        base_score = 1.0
        event_penalty = min(0.5, self.rate_limit_events * 0.1)
        
        # Recent failure penalty
        if self.last_quota_failure:
            hours_since_failure = (datetime.now(timezone.utc) - self.last_quota_failure).total_seconds() / 3600
            failure_penalty = max(0.0, 0.3 - (hours_since_failure * 0.1))
        else:
            failure_penalty = 0.0
        
        return max(0.0, base_score - event_penalty - failure_penalty)
    
    def should_skip_for_quota(self) -> bool:
        """Determine if provider should be skipped due to quota concerns."""
        return not self.is_available()
    
    def get_retry_after_seconds(self) -> Optional[int]:
        """Get suggested retry-after seconds if rate limited."""
        if self.availability == ProviderAvailability.COOLDOWN and self.cooldown_until:
            remaining = (self.cooldown_until - datetime.now(timezone.utc)).total_seconds()
            return max(1, int(remaining))
        
        if self.last_429_response and "retry_after" in self.last_429_response:
            return self.last_429_response["retry_after"]
        
        return None