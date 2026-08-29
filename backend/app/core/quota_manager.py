from typing import Optional, Dict, List, Any
from datetime import datetime, timezone

from app.models.quota import (
    ProviderQuotaState, 
    ProviderAvailability, 
    QuotaConfidence,
    QuotaResetInfo
)
from app.core.quota_storage import get_quota_storage, QuotaStorageBackend
from app.core.logging import get_logger

logger = get_logger(__name__)


class QuotaManager:
    """
    Manages provider quota state, cooldown periods, and rate limit detection.
    Provides routing decisions based on quota health.
    """
    
    def __init__(self, storage: Optional[QuotaStorageBackend] = None):
        """
        Initialize quota manager.
        
        Args:
            storage: Optional custom storage backend. If None, uses global instance.
        """
        self.storage = storage or get_quota_storage()
        self._cooldown_config = {
            "default_cooldown_seconds": 60,
            "exhausted_cooldown_seconds": 300,  # 5 minutes for exhausted state
            "max_consecutive_failures": 3,
            "health_check_interval_seconds": 30
        }
    
    async def get_provider_state(self, provider_name: str) -> ProviderQuotaState:
        """
        Get quota state for a provider, creating default if not exists.
        
        Args:
            provider_name: Name of the provider
            
        Returns:
            ProviderQuotaState instance
        """
        state = await self.storage.get_provider_state(provider_name)
        if state is None:
            state = ProviderQuotaState(provider_name=provider_name)
            await self.storage.save_provider_state(state)
            logger.debug(f"Created default quota state for provider: {provider_name}")
        return state
    
    async def record_request(self, provider_name: str, tokens_used: int = 0) -> None:
        """
        Record a successful request for a provider.
        
        Args:
            provider_name: Name of the provider
            tokens_used: Number of tokens consumed in the request
        """
        state = await self.get_provider_state(provider_name)
        state.record_request(tokens_used)
        await self.storage.save_provider_state(state)
        logger.debug(f"Recorded request for provider {provider_name}: tokens={tokens_used}")
    
    async def record_rate_limit(
        self, 
        provider_name: str, 
        status_code: int = 429,
        response_headers: Optional[Dict[str, str]] = None,
        response_body: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record a rate limit event for a provider.
        
        Args:
            provider_name: Name of the provider
            status_code: HTTP status code (default 429)
            response_headers: Response headers for quota info extraction
            response_body: Response body for quota info extraction
        """
        state = await self.get_provider_state(provider_name)
        
        # Extract retry-after from headers
        retry_after = None
        if response_headers:
            retry_after = self._extract_retry_after(response_headers)
        
        # Extract quota reset info from response
        reset_info = self._extract_reset_info(response_headers, response_body)
        if reset_info:
            state.update_reset_info(reset_info)
        
        # Determine cooldown duration
        cooldown_seconds = self._determine_cooldown_duration(state, retry_after)
        
        # Record the rate limit event
        response_details = {
            "status_code": status_code,
            "headers": response_headers,
            "retry_after": retry_after,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        state.record_rate_limit(response_details, cooldown_seconds)
        await self.storage.save_provider_state(state)
        
        logger.warning(
            f"Rate limit recorded for provider {provider_name}: "
            f"state={state.availability.value}, "
            f"consecutive_failures={state.consecutive_failures}, "
            f"cooldown_until={state.cooldown_until}"
        )
    
    def _extract_retry_after(self, headers: Dict[str, str]) -> Optional[int]:
        """Extract retry-after seconds from headers."""
        retry_after = headers.get("Retry-After") or headers.get("retry-after")
        if retry_after:
            try:
                return int(retry_after)
            except ValueError:
                pass
        return None
    
    def _extract_reset_info(
        self, 
        headers: Optional[Dict[str, str]], 
        body: Optional[Dict[str, Any]]
    ) -> Optional[QuotaResetInfo]:
        """
        Extract quota reset information from provider response.
        Different providers may expose this in different formats.
        """
        reset_info = None
        
        # Try to extract from headers first
        if headers:
            # Common patterns for reset time
            reset_time = headers.get("X-RateLimit-Reset") or headers.get("x-ratelimit-reset")
            remaining = headers.get("X-RateLimit-Remaining") or headers.get("x-ratelimit-remaining")
            
            if reset_time or remaining:
                reset_info = QuotaResetInfo(
                    reset_time=datetime.fromtimestamp(int(reset_time), timezone.utc) if reset_time else None,
                    requests_remaining=int(remaining) if remaining else None
                )
        
        # Try to extract from body if not found in headers
        if not reset_info and body:
            # Common patterns in JSON responses
            if "quota" in body:
                quota_data = body["quota"]
                reset_info = QuotaResetInfo(
                    reset_time=datetime.fromtimestamp(quota_data.get("reset", 0), timezone.utc) if quota_data.get("reset") else None,
                    requests_remaining=quota_data.get("remaining"),
                    tokens_remaining=quota_data.get("tokens_remaining")
                )
        
        return reset_info
    
    def _determine_cooldown_duration(self, state: ProviderQuotaState, retry_after: Optional[int]) -> int:
        """Determine appropriate cooldown duration based on state and response."""
        # Use provider-specified retry-after if available
        if retry_after:
            return retry_after
        
        # Use state-based cooldown
        if state.consecutive_failures >= self._cooldown_config["max_consecutive_failures"]:
            return self._cooldown_config["exhausted_cooldown_seconds"]
        
        return self._cooldown_config["default_cooldown_seconds"]
    
    async def is_provider_available(self, provider_name: str) -> bool:
        """
        Check if a provider is available for requests.
        
        Args:
            provider_name: Name of the provider
            
        Returns:
            True if available, False otherwise
        """
        state = await self.get_provider_state(provider_name)
        return state.is_available()
    
    async def get_available_providers(self, provider_names: List[str]) -> List[str]:
        """
        Get list of available providers from a candidate list.
        
        Args:
            provider_names: List of provider names to check
            
        Returns:
            List of available provider names
        """
        available = []
        for provider_name in provider_names:
            if await self.is_provider_available(provider_name):
                available.append(provider_name)
        return available
    
    async def get_provider_health_score(self, provider_name: str) -> float:
        """
        Get health score for a provider (0.0 to 1.0).
        
        Args:
            provider_name: Name of the provider
            
        Returns:
            Health score (higher is better)
        """
        state = await self.get_provider_state(provider_name)
        return state.get_estimated_quota_health()
    
    async def get_provider_health_scores(self, provider_names: List[str]) -> Dict[str, float]:
        """
        Get health scores for multiple providers.
        
        Args:
            provider_names: List of provider names
            
        Returns:
            Dictionary mapping provider names to health scores
        """
        scores = {}
        for provider_name in provider_names:
            scores[provider_name] = await self.get_provider_health_score(provider_name)
        return scores
    
    async def reset_provider_state(self, provider_name: str) -> None:
        """
        Reset a provider's quota state to available.
        Useful for manual recovery or testing.
        
        Args:
            provider_name: Name of the provider
        """
        state = await self.get_provider_state(provider_name)
        state.availability = ProviderAvailability.AVAILABLE
        state.cooldown_until = None
        state.consecutive_failures = 0
        state.last_quota_failure = None
        state.last_429_response = None
        await self.storage.save_provider_state(state)
        logger.info(f"Reset quota state for provider: {provider_name}")
    
    async def get_all_states(self) -> Dict[str, ProviderQuotaState]:
        """
        Get all provider states.
        
        Returns:
            Dictionary mapping provider names to quota states
        """
        return await self.storage.get_all_states()
    
    async def clear_all_states(self) -> None:
        """Clear all provider states."""
        await self.storage.clear_all_states()
        logger.info("Cleared all quota states")
    
    async def cleanup_expired_states(self) -> int:
        """
        Clean up expired cooldown states and mark providers as available.
        
        Returns:
            Number of states that were updated
        """
        states = await self.storage.get_all_states()
        updated_count = 0
        
        for provider_name, state in states.items():
            if state.availability == ProviderAvailability.COOLDOWN:
                if state.is_available():  # This checks if cooldown has expired
                    await self.storage.save_provider_state(state)
                    updated_count += 1
                    logger.info(f"Auto-recovered provider from cooldown: {provider_name}")
        
        return updated_count
    
    async def get_quota_status_for_routing(self, provider_name: str) -> Dict[str, Any]:
        """
        Get quota status information for routing decisions.
        
        Args:
            provider_name: Name of the provider
            
        Returns:
            Dictionary with routing-relevant quota information
        """
        state = await self.get_provider_state(provider_name)
        
        return {
            "provider_name": provider_name,
            "available": state.is_available(),
            "availability": state.availability.value,
            "health_score": state.get_estimated_quota_health(),
            "quota_confidence": state.quota_confidence.value,
            "request_count": state.request_count,
            "token_usage": state.token_usage,
            "rate_limit_events": state.rate_limit_events,
            "consecutive_failures": state.consecutive_failures,
            "cooldown_until": state.cooldown_until.isoformat() if state.cooldown_until else None,
            "retry_after": state.get_retry_after_seconds(),
            "reset_info": state.reset_info.model_dump() if state.reset_info else None
        }


# Global quota manager instance
_quota_manager: Optional[QuotaManager] = None


def get_quota_manager() -> QuotaManager:
    """Get the global quota manager instance."""
    global _quota_manager
    if _quota_manager is None:
        _quota_manager = QuotaManager()
    return _quota_manager


def set_quota_manager(manager: QuotaManager) -> None:
    """Set the global quota manager instance (for testing)."""
    global _quota_manager
    _quota_manager = manager