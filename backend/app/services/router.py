from typing import Optional
from app.models.chat import ChatCompletionRequest, ChatCompletionResponse
from app.providers.config_registry import configurable_registry
from app.providers.errors import ProviderUnsupportedModelError

class RoutingEngine:
    """
    The core routing engine that determines which provider to use.
    Consumes the configurable provider registry dynamically without hardcoded provider names.
    """
    
    def __init__(self, registry: Optional[object] = None):
        """
        Initialize routing engine with a provider registry.
        
        Args:
            registry: Provider registry instance. If None, uses global configurable_registry.
        """
        self.registry = registry or configurable_registry
    
    async def route_chat_completion(self, request: ChatCompletionRequest, target_provider: Optional[str] = None) -> ChatCompletionResponse:
        """
        Routes the chat completion to a provider.
        
        Args:
            request: Chat completion request
            target_provider: Specific provider name to use. If None, selects based on priority.
            
        Returns:
            Chat completion response
            
        Raises:
            RuntimeError: If no providers are available
            ProviderUnsupportedModelError: If model is not supported by the provider
        """
        # Select provider based on priority if not specified
        if not target_provider:
            providers_by_priority = self.registry.get_provider_by_priority()
            if not providers_by_priority:
                raise RuntimeError("No providers registered in AIFlow.")
            target_provider = providers_by_priority[0]
            
        provider = self.registry.get_provider(target_provider)
        
        # Verify model is supported if specified
        if request.model != "default" and request.model not in provider.get_supported_models():
            raise ProviderUnsupportedModelError(f"Model '{request.model}' not supported by provider '{target_provider}'")
        
        # Track active request for hot-reload safety
        await self.registry.increment_active_requests()
        try:
            # Delegate the call
            return await provider.send_chat_completion(request)
        finally:
            await self.registry.decrement_active_requests()
    
    def get_available_providers(self) -> list:
        """Get list of available provider names."""
        return self.registry.list_providers()
    
    def get_provider_config(self, provider_name: str):
        """Get configuration for a specific provider."""
        return self.registry.get_provider_config(provider_name)
    
    def get_providers_by_capability(self, capability: str) -> list:
        """Get providers that support a specific capability."""
        return self.registry.get_providers_by_capability(capability)

# Singleton routing engine using configurable registry
router_engine = RoutingEngine()
