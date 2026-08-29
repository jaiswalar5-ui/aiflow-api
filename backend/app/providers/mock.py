import time
import uuid
from typing import List, Type, Optional
from app.providers.base import ProviderBase
from app.providers.errors import ProviderError
from app.models.chat import (
    ChatCompletionRequest, ChatCompletionResponse, 
    ChatChoice, ChatMessage, ChatCompletionUsage
)

class MockProvider(ProviderBase):
    """
    A mock provider that always returns a successful predefined response.
    """
    def __init__(self, timeout_seconds: float = 30.0, supported_models: Optional[List[str]] = None, api_key: Optional[str] = None):
        self.timeout = timeout_seconds
        self.api_key = api_key
        # Use provided models or default, but don't override if provided
        if supported_models is not None:
            self._supported_models = supported_models
        else:
            self._supported_models = ["mock-gpt", "mock-claude"]
        
    def get_supported_models(self) -> List[str]:
        return self._supported_models
        
    async def check_health(self) -> bool:
        return True
        
    async def send_chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        message = ChatMessage(role="assistant", content="This is a mock provider response.")
        choice = ChatChoice(index=0, message=message, finish_reason="stop")
        usage = ChatCompletionUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20)
        
        from app.models.chat import ResponseMetadata
        return ChatCompletionResponse(
            id=f"mock-{uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model=request.model,
            provider="mock",
            choices=[choice],
            usage=usage,
            metadata=ResponseMetadata(
                provider_name="mock",
                provider_model=request.model
            )
        )

class MockFailingProvider(ProviderBase):
    """
    A mock provider that always raises a specified ProviderError.
    """
    def __init__(self, error_to_raise: Exception, supported_models: Optional[List[str]] = None):
        self._error = error_to_raise
        self._supported_models = supported_models or ["mock-failing"]
        
    def get_supported_models(self) -> List[str]:
        return self._supported_models
        
    async def check_health(self) -> bool:
        return False
        
    async def send_chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        raise self._error
