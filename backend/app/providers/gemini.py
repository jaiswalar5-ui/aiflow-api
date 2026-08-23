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

class GeminiProvider(ProviderBase):
    """
    Provider adapter for Gemini REST API via httpx.
    """
    def __init__(self, timeout_seconds: float = 30.0):
        self.timeout = timeout_seconds
        self.api_key = settings.GEMINI_API_KEY
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"
        self.supported_models = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro"]

    def get_supported_models(self) -> List[str]:
        return self.supported_models

    async def check_health(self) -> bool:
        if not self.api_key:
            return False
        # A lightweight way to check health is getting a model list, but for now we'll assume true if key exists
        # In a real app we might do a GET /models with the key
        return True

    def _translate_request(self, request: ChatCompletionRequest) -> dict:
        """Translates canonical format to Gemini Content format."""
        contents = []
        for msg in request.messages:
            # Gemini roles: "user" or "model"
            role = "user" if msg.role == "user" else "model"
            # Note: "system" role usually requires special handling in Gemini via `systemInstruction`
            # For MVP, we treat "system" as "user" or ignore. We'll map "system" to "user" for now,
            # or handle properly if supported.
            if msg.role == "system":
                role = "user"
            
            contents.append({
                "role": role,
                "parts": [{"text": msg.content}]
            })
            
        payload: dict = {"contents": contents}
        
        # Generation config
        gen_config = {}
        if request.temperature is not None:
            gen_config["temperature"] = request.temperature
        if request.max_tokens is not None:
            gen_config["maxOutputTokens"] = request.max_tokens
        if request.top_p is not None:
            gen_config["topP"] = request.top_p
            
        if gen_config:
            payload["generationConfig"] = gen_config
            
        return payload

    def _convert_response(self, response_data: dict, model_id: str) -> ChatCompletionResponse:
        """Converts Gemini response to canonical format."""
        candidates = response_data.get("candidates", [])
        if not candidates:
            raise ProviderUnknownError("No candidates returned from Gemini")
            
        candidate = candidates[0]
        content_parts = candidate.get("content", {}).get("parts", [])
        text = content_parts[0].get("text", "") if content_parts else ""
        
        finish_reason = candidate.get("finishReason", "stop").lower()
        
        choice = ChatChoice(
            index=0,
            message=ChatMessage(role="assistant", content=text),
            finish_reason=finish_reason
        )
        
        usage_meta = response_data.get("usageMetadata", {})
        usage = ChatCompletionUsage(
            prompt_tokens=usage_meta.get("promptTokenCount", 0),
            completion_tokens=usage_meta.get("candidatesTokenCount", 0),
            total_tokens=usage_meta.get("totalTokenCount", 0),
        )
        
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model=model_id,
            choices=[choice],
            usage=usage
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
            raise ProviderUnsupportedModelError(f"Model not found: {exc.response.text}") from exc
        elif status in (500, 502, 503, 504):
            raise ProviderServerError(f"Server error: {exc.response.text}") from exc
        else:
            raise ProviderUnknownError(f"Unknown HTTP error {status}: {exc.response.text}") from exc

    async def send_chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        if not self.api_key:
            raise ProviderAuthenticationError("GEMINI_API_KEY is not configured.")
            
        model = request.model
        # Optional: default mapping
        if model == "default":
            model = "gemini-1.5-flash"
            
        if model not in self.supported_models:
            raise ProviderUnsupportedModelError(f"Model {model} not supported by GeminiProvider.")
            
        url = f"{self.base_url}/{model}:generateContent?key={self.api_key}"
        payload = self._translate_request(request)
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                return self._convert_response(response.json(), model)
                
        except httpx.HTTPStatusError as exc:
            self._handle_http_error(exc)
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError("Gemini API request timed out") from exc
        except httpx.RequestError as exc:
            raise ProviderNetworkError(f"Network error connecting to Gemini: {str(exc)}") from exc
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderUnknownError(f"Unexpected error: {str(exc)}") from exc
