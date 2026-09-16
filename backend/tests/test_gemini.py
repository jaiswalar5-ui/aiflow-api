import pytest
import respx
import httpx
from app.providers.gemini import GeminiProvider
from app.providers.errors import (
    ProviderAuthenticationError, ProviderRateLimitError,
    ProviderInvalidRequestError, ProviderUnsupportedModelError,
    ProviderServerError, ProviderTimeoutError, ProviderNetworkError,
    ProviderUnknownError, QuotaExhaustedError
)
from app.models.chat import ChatCompletionRequest, ChatMessage

@pytest.fixture
def gemini_provider(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-fake-key")
    from app.core.config import settings
    settings.GEMINI_API_KEY = "test-fake-key"
    return GeminiProvider(timeout_seconds=1.0, api_key="test-fake-key")

@pytest.fixture
def basic_request():
    return ChatCompletionRequest(
        model="gemini-1.5-flash",
        messages=[ChatMessage(role="user", content="Hello Gemini")]
    )

@pytest.mark.asyncio
@respx.mock
async def test_gemini_success(gemini_provider, basic_request):
    mock_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=test-fake-key"
    
    mock_response = {
        "candidates": [{
            "content": {"parts": [{"text": "Hello User!"}]},
            "finishReason": "STOP"
        }],
        "usageMetadata": {
            "promptTokenCount": 5,
            "candidatesTokenCount": 3,
            "totalTokenCount": 8
        }
    }
    
    respx.post(mock_url).mock(return_value=httpx.Response(200, json=mock_response))
    
    response = await gemini_provider.send_chat_completion(basic_request)
    
    assert response.model == "gemini-1.5-flash"
    assert len(response.choices) == 1
    assert response.choices[0].message.content == "Hello User!"
    assert response.choices[0].finish_reason == "stop"
    assert response.usage.total_tokens == 8

@pytest.mark.asyncio
@respx.mock
async def test_gemini_auth_error(gemini_provider, basic_request):
    respx.post().mock(return_value=httpx.Response(401, json={"error": "Unauthorized"}))
    with pytest.raises(ProviderAuthenticationError):
        await gemini_provider.send_chat_completion(basic_request)

@pytest.mark.asyncio
@respx.mock
async def test_gemini_rate_limit(gemini_provider, basic_request):
    respx.post().mock(return_value=httpx.Response(
        429,
        json={"error": {"code": "RATE_LIMIT_EXCEEDED", "message": "Rate limit exceeded"}}
    ))
    with pytest.raises(ProviderRateLimitError):
        await gemini_provider.send_chat_completion(basic_request)

@pytest.mark.asyncio
@respx.mock
async def test_gemini_quota_exhausted(gemini_provider, basic_request):
    respx.post().mock(return_value=httpx.Response(
        429,
        json={"error": {"code": "QUOTA_EXCEEDED", "message": "Quota exceeded"}}
    ))
    with pytest.raises(QuotaExhaustedError):
        await gemini_provider.send_chat_completion(basic_request)

@pytest.mark.asyncio
@respx.mock
async def test_gemini_invalid_request(gemini_provider, basic_request):
    respx.post().mock(return_value=httpx.Response(400, json={"error": "Bad Request"}))
    with pytest.raises(ProviderInvalidRequestError):
        await gemini_provider.send_chat_completion(basic_request)

@pytest.mark.asyncio
@respx.mock
async def test_gemini_server_error(gemini_provider, basic_request):
    respx.post().mock(return_value=httpx.Response(500, json={"error": "Internal Server Error"}))
    with pytest.raises(ProviderServerError):
        await gemini_provider.send_chat_completion(basic_request)

@pytest.mark.asyncio
@respx.mock
async def test_gemini_timeout(gemini_provider, basic_request):
    respx.post().mock(side_effect=httpx.TimeoutException("Timeout"))
    with pytest.raises(ProviderTimeoutError):
        await gemini_provider.send_chat_completion(basic_request)

@pytest.mark.asyncio
@respx.mock
async def test_gemini_network_error(gemini_provider, basic_request):
    respx.post().mock(side_effect=httpx.NetworkError("Network issue"))
    with pytest.raises(ProviderNetworkError):
        await gemini_provider.send_chat_completion(basic_request)

@pytest.mark.asyncio
async def test_gemini_missing_api_key(basic_request):
    provider = GeminiProvider()
    provider.api_key = None
    with pytest.raises(ProviderAuthenticationError):
        await provider.send_chat_completion(basic_request)

@pytest.mark.asyncio
async def test_gemini_unsupported_model(gemini_provider):
    req = ChatCompletionRequest(model="gpt-4", messages=[ChatMessage(role="user", content="Hi")])
    with pytest.raises(ProviderUnsupportedModelError):
        await gemini_provider.send_chat_completion(req)
