import time
import uuid
import httpx
from typing import List, Optional

from app.core.config import settings
from app.providers.base import ProviderBase
from app.providers.errors import (
    ProviderError,
    AuthenticationError,
    RateLimitError,
    TimeoutError,
    ServerError,
    InvalidRequestError,
    UnsupportedModelError,
    NetworkError,
    UnknownError
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
    def __init__(self, timeout_seconds: float = 30.0, api_key: Optional[str] = None):
        self.timeout = timeout_seconds
        self.api_key = api_key or settings.GEMINI_API_KEY
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
        
        from app.models.chat import ResponseMetadata
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model=model_id,
            provider="gemini",
            choices=[choice],
            usage=usage,
            metadata=ResponseMetadata(
                provider_name="gemini",
                provider_model=model_id
            )
        )

    def _handle_http_error(self, exc: httpx.HTTPStatusError):
        """Use centralized error classifier to map HTTP errors to standardized provider errors."""
        from app.core.error_classifier import classify_error, get_error_classifier
        
        status = exc.response.status_code
        headers = dict(exc.response.headers)
        
        try:
            body = exc.response.json()
            raw_body = exc.response.text
        except:
            body = {"text": exc.response.text}
            raw_body = exc.response.text
        
        # Use centralized classifier
        classification = classify_error(
            provider_name="gemini",
            http_status=status,
            raw_error_body=raw_body,
            response_headers=headers,
            original_error=exc.response.text
        )
        
        # Create appropriate error using classification result
        classifier = get_error_classifier()
        provider_error = classifier.create_provider_error(
            classification=classification,
            provider_name="gemini",
            request_id=None
        )
        
        raise provider_error from exc

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
            from app.core.error_classifier import classify_error
            classification = classify_error(
                provider_name="gemini",
                http_status=None,
                raw_error_body=str(exc),
                response_headers=None,
                original_error="Request timeout"
            )
            classifier = get_error_classifier()
            provider_error = classifier.create_provider_error(
                classification=classification,
                provider_name="gemini",
                request_id=None
            )
            raise provider_error from exc
        except httpx.RequestError as exc:
            from app.core.error_classifier import classify_error
            classification = classify_error(
                provider_name="gemini",
                http_status=None,
                raw_error_body=str(exc),
                response_headers=None,
                original_error="Network error"
            )
            classifier = get_error_classifier()
            provider_error = classifier.create_provider_error(
                classification=classification,
                provider_name="gemini",
                request_id=None
            )
            raise provider_error from exc
        except ProviderError:
            raise
        except Exception as exc:
            from app.core.error_classifier import classify_error
            classification = classify_error(
                provider_name="gemini",
                http_status=None,
                raw_error_body=str(exc),
                response_headers=None,
                original_error="Unexpected error"
            )
            classifier = get_error_classifier()
            provider_error = classifier.create_provider_error(
                classification=classification,
                provider_name="gemini",
                request_id=None
            )
            raise provider_error from exc
