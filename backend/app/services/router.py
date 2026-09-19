from typing import Optional
from app.models.chat import ChatCompletionRequest, ChatCompletionResponse
from app.providers.config_registry import configurable_registry
from app.providers.errors import ProviderUnsupportedModelError, ProviderRateLimitError
from app.core.quota_manager import get_quota_manager
from app.core.logging import get_logger

logger = get_logger(__name__)

class RoutingEngine:
    """
    The core routing engine that determines which provider to use.
    Consumes the configurable provider registry dynamically without hardcoded provider names.
    Integrates with quota manager for intelligent routing decisions.
    """
    
    def __init__(self, registry=None, quota_manager=None):
        """
        Initialize routing engine with a provider registry and quota manager.
        """
        self.registry = registry or configurable_registry
        self.quota_manager = quota_manager or get_quota_manager()
    
    async def route_chat_completion(self, request: ChatCompletionRequest, target_provider: Optional[str] = None) -> ChatCompletionResponse:
        """
        Routes the chat completion to a provider with quota-aware routing.
        
        Args:
            request: Chat completion request
            target_provider: Specific provider name to use. If None, selects based on priority and quota.
            
        Returns:
            Chat completion response
            
        Raises:
            RuntimeError: If no providers are available
            ProviderUnsupportedModelError: If model is not supported by the provider
        """
        # Select provider based on priority and quota if not specified
        if not target_provider:
            target_provider = await self._select_best_provider(request)
            if not target_provider:
                raise RuntimeError("No available providers in AIFlow.")
            
        provider = self.registry.get_provider(target_provider)
        
        # Verify model is supported if specified
        if request.model != "default" and request.model not in provider.get_supported_models():
            raise ProviderUnsupportedModelError(f"Model '{request.model}' not supported by provider '{target_provider}'")
        
        # Track active request for hot-reload safety
        await self.registry.increment_active_requests()
        try:
            # Delegate the call with the retry engine
            from app.api.dependencies import get_retry_engine
            retry_engine = get_retry_engine()
            
            response = await retry_engine.execute_with_retry(
                provider.send_chat_completion,
                request
            )
            
            # Record successful request in quota manager
            tokens_used = (response.usage.total_tokens or 0) if response.usage else 0
            await self.quota_manager.record_request(target_provider, tokens_used)
            
            return response
        except ProviderRateLimitError as e:
            # Record rate limit event
            await self.quota_manager.record_rate_limit(
                target_provider,
                status_code=e.status_code or 429,
                response_headers=e.response_headers,
                response_body=e.response_body
            )
            logger.warning(f"Rate limit hit for provider {target_provider}: {e}")
            raise
        finally:
            await self.registry.decrement_active_requests()
    
    async def _select_best_provider(self, request: ChatCompletionRequest) -> Optional[str]:
        """
        Select the best provider based on priority, quota health, and model support.
        
        Args:
            request: Chat completion request
            
        Returns:
            Best provider name or None if no suitable provider found
        """
        providers_by_priority = self.registry.get_provider_by_priority()
        if not providers_by_priority:
            return None
        
        # Get available providers (not in cooldown/exhausted)
        available_providers = await self.quota_manager.get_available_providers(providers_by_priority)
        
        if not available_providers:
            logger.warning("All providers are in cooldown or exhausted. Using highest priority provider.")
            return providers_by_priority[0]
        
        # Filter providers that support the requested model
        model = request.model if request.model != "default" else None
        if model:
            model_supported = []
            for provider_name in available_providers:
                provider = self.registry.get_provider(provider_name)
                if model in provider.get_supported_models():
                    model_supported.append(provider_name)
            candidates = model_supported if model_supported else available_providers
        else:
            candidates = available_providers
        
        if not candidates:
            # No provider supports the model, use highest priority available
            return available_providers[0]
        
        # Get health scores for candidates
        health_scores = await self.quota_manager.get_provider_health_scores(candidates)
        
        # Select provider with highest health score
        best_provider = max(candidates, key=lambda p: health_scores.get(p, 0.5))
        
        logger.debug(f"Selected provider {best_provider} with health score {health_scores.get(best_provider, 0.5)}")
        return best_provider
    
    def get_available_providers(self) -> list:
        """Get list of available provider names."""
        return self.registry.list_providers()
    
    def get_provider_config(self, provider_name: str):
        """Get configuration for a specific provider."""
        return self.registry.get_provider_config(provider_name)
    
    def get_providers_by_capability(self, capability: str) -> list:
        """Get providers that support a specific capability."""
        return self.registry.get_providers_by_capability(capability)
    
    async def get_quota_status(self, provider_name: str) -> dict:
        """Get quota status for a specific provider."""
        return await self.quota_manager.get_quota_status_for_routing(provider_name)
    
    async def get_all_quota_status(self) -> dict:
        """Get quota status for all providers."""
        providers = self.registry.list_providers()
        status = {}
        for provider_name in providers:
            status[provider_name] = await self.quota_manager.get_quota_status_for_routing(provider_name)
        return status

# Singleton routing engine using configurable registry and quota manager
router_engine = RoutingEngine()
