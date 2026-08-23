from typing import Optional
from app.models.chat import ChatCompletionRequest, ChatCompletionResponse
from app.providers.registry import provider_registry
from app.providers.errors import ProviderUnsupportedModelError

class RoutingEngine:
    """
    The core routing engine that determines which provider to use.
    For this initial phase, it just accepts a requested provider name and delegates the call.
    """
    
    async def route_chat_completion(self, request: ChatCompletionRequest, target_provider: Optional[str] = None) -> ChatCompletionResponse:
        """
        Routes the chat completion to a provider.
        """
        # In the future, this will intelligently select a provider based on cost, latency, or requested model.
        # For MVP abstraction, we expect target_provider to be passed or default to a registered one.
        
        if not target_provider:
            providers = provider_registry.list_providers()
            if not providers:
                raise RuntimeError("No providers registered in AIFlow.")
            target_provider = providers[0]
            
        provider = provider_registry.get_provider(target_provider)
        
        # Verify model is supported if specified
        if request.model != "default" and request.model not in provider.get_supported_models():
            raise ProviderUnsupportedModelError(f"Model '{request.model}' not supported by provider '{target_provider}'")
            
        # Delegate the call
        return await provider.send_chat_completion(request)

# Singleton routing engine
router_engine = RoutingEngine()
