from abc import ABC, abstractmethod
from typing import List
from app.models.chat import ChatCompletionRequest, ChatCompletionResponse

class ProviderBase(ABC):
    """
    Abstract base class for all AI providers (OpenAI, Anthropic, Gemini, etc.).
    """
    
    @abstractmethod
    async def send_chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """
        Send a chat completion request to the provider.
        Must raise appropriate exceptions from app.providers.errors on failure.
        """
        pass
        
    @abstractmethod
    def get_supported_models(self) -> List[str]:
        """
        Return a list of models supported by this provider adapter.
        """
        pass
        
    @abstractmethod
    async def check_health(self) -> bool:
        """
        Check if the provider is currently accessible/healthy.
        """
        pass
