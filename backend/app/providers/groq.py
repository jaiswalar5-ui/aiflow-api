import time
import uuid
import httpx
from typing import List, Optional

from app.core.config import settings
from app.providers.base import ProviderBase
from app.providers.errors import (
    ProviderError,
    ProviderAuthenticationError,
    ProviderRateLimitError,
    ProviderInvalidRequestError,
    ProviderUnsupportedModelError,
    ProviderServerError,
    ProviderTimeoutError,
    ProviderNetworkError,
    ProviderUnknownError,
)
from app.models.chat import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatChoice,
    ChatMessage,
    ChatCompletionUsage,
)

class GroqProvider(ProviderBase):
    """
    Provider adapter for Groq REST API via httpx.
    """
    def __init__(self, timeout_seconds: float = 30.0, api_key: Optional[str] = None):
        self.timeout = timeout_seconds
        self.api_key = api_key or settings.GROQ_API_KEY
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.supported_models = [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "gemma2-9b-it"
        ]

    def get_supported_models(self) -> List[str]:
        return self.supported_models

    async def check_health(self) -> bool:
        if not self.api_key:
            return False
        return True

    def _translate_request(self, request: ChatCompletionRequest, model_id: str) -> dict:
        """Translates canonical format to Groq OpenAI-compatible format."""
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
            
        payload: dict = {
            "model": model_id,
            "messages": messages
        }
        
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.top_p is not None:
            payload["top_p"] = request.top_p
            
        return payload

    def _convert_response(self, response_data: dict, model_id: str) -> ChatCompletionResponse:
        """Converts Groq response to canonical format."""
        choices_data = response_data.get("choices", [])
        if not choices_data:
            raise ProviderUnknownError("No choices returned from Groq")
            
        choice_data = choices_data[0]
        message_data = choice_data.get("message", {})
        
        choice = ChatChoice(
            index=choice_data.get("index", 0),
            message=ChatMessage(
                role=message_data.get("role", "assistant"),
                content=message_data.get("content", "")
            ),
            finish_reason=choice_data.get("finish_reason", "stop")
        )
        
        usage_data = response_data.get("usage", {})
        usage = ChatCompletionUsage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )
        
        from app.models.chat import ResponseMetadata
        return ChatCompletionResponse(
            id=response_data.get("id", f"chatcmpl-{uuid.uuid4().hex[:12]}"),
            created=response_data.get("created", int(time.time())),
            model=response_data.get("model", model_id),
            provider="groq",
            choices=[choice],
            usage=usage,
            metadata=ResponseMetadata(
                provider_name="groq",
                provider_model=response_data.get("model", model_id)
            )
        )

    def _handle_http_error(self, exc: httpx.HTTPStatusError):
        """Maps HTTP status codes to standardized provider errors."""
        status = exc.response.status_code
        if status in (401, 403):
            raise ProviderAuthenticationError(f"Authentication failed: {exc.response.text}") from exc
        elif status == 429:
            raise ProviderRateLimitError(f"Rate limit exceeded: {exc.response.text}") from exc
        elif status == 400:
            raise ProviderInvalidRequestError(f"Invalid request: {exc.response.text}") from exc
        elif status == 404:
            raise ProviderUnsupportedModelError(f"Model or endpoint not found: {exc.response.text}") from exc
        elif status in (500, 502, 503, 504):
            raise ProviderServerError(f"Server error: {exc.response.text}") from exc
        else:
            raise ProviderUnknownError(f"Unknown HTTP error {status}: {exc.response.text}") from exc

    async def send_chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        if not self.api_key:
            raise ProviderAuthenticationError("GROQ_API_KEY is not configured.")
            
        model = request.model
        if model == "default":
            model = "llama-3.1-8b-instant"
            
        if model not in self.supported_models:
            raise ProviderUnsupportedModelError(f"Model {model} not supported by GroqProvider.")
            
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = self._translate_request(request, model)
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(self.base_url, headers=headers, json=payload)
                response.raise_for_status()
                return self._convert_response(response.json(), model)
                
        except httpx.HTTPStatusError as exc:
            self._handle_http_error(exc)
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError("Groq API request timed out") from exc
        except httpx.RequestError as exc:
            raise ProviderNetworkError(f"Network error connecting to Groq: {str(exc)}") from exc
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderUnknownError(f"Unexpected error: {str(exc)}") from exc
