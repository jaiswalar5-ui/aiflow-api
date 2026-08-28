import pytest
import respx
import httpx
from app.providers.groq import GroqProvider
from app.providers.errors import (
    ProviderAuthenticationError, ProviderRateLimitError,
    ProviderInvalidRequestError, ProviderUnsupportedModelError,
    ProviderServerError, ProviderTimeoutError, ProviderNetworkError,
    ProviderUnknownError
)
from app.models.chat import ChatCompletionRequest, ChatMessage

@pytest.fixture
def groq_provider(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test-fake-key")
    from app.core.config import settings
    settings.GROQ_API_KEY = "gsk_test-fake-key"
    return GroqProvider(timeout_seconds=1.0, api_key="gsk_test-fake-key")

@pytest.fixture
def basic_request():
    return ChatCompletionRequest(
        model="llama-3.1-8b-instant",
        messages=[ChatMessage(role="user", content="Hello Groq")]
    )

@pytest.mark.asyncio
@respx.mock
async def test_groq_success(groq_provider, basic_request):
    mock_url = "https://api.groq.com/openai/v1/chat/completions"
    
    mock_response = {
        "id": "chatcmpl-12345",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "llama-3.1-8b-instant",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello User!"
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 9,
            "completion_tokens": 12,
            "total_tokens": 21
        }
    }
    
    respx.post(mock_url).mock(return_value=httpx.Response(200, json=mock_response))
    
    response = await groq_provider.send_chat_completion(basic_request)
    
    assert response.model == "llama-3.1-8b-instant"
    assert len(response.choices) == 1
    assert response.choices[0].message.content == "Hello User!"
    assert response.choices[0].finish_reason == "stop"
    assert response.usage.total_tokens == 21

@pytest.mark.asyncio
@respx.mock
async def test_groq_auth_error(groq_provider, basic_request):
    respx.post().mock(return_value=httpx.Response(401, json={"error": "Unauthorized"}))
    with pytest.raises(ProviderAuthenticationError):
        await groq_provider.send_chat_completion(basic_request)

@pytest.mark.asyncio
@respx.mock
async def test_groq_rate_limit(groq_provider, basic_request):
    respx.post().mock(return_value=httpx.Response(429, json={"error": "Rate limit exceeded"}))
    with pytest.raises(ProviderRateLimitError):
        await groq_provider.send_chat_completion(basic_request)

@pytest.mark.asyncio
@respx.mock
async def test_groq_invalid_request(groq_provider, basic_request):
    respx.post().mock(return_value=httpx.Response(400, json={"error": "Bad Request"}))
    with pytest.raises(ProviderInvalidRequestError):
        await groq_provider.send_chat_completion(basic_request)

@pytest.mark.asyncio
@respx.mock
async def test_groq_server_error(groq_provider, basic_request):
    respx.post().mock(return_value=httpx.Response(500, json={"error": "Internal Server Error"}))
    with pytest.raises(ProviderServerError):
        await groq_provider.send_chat_completion(basic_request)

@pytest.mark.asyncio
@respx.mock
async def test_groq_timeout(groq_provider, basic_request):
    respx.post().mock(side_effect=httpx.TimeoutException("Timeout"))
    with pytest.raises(ProviderTimeoutError):
        await groq_provider.send_chat_completion(basic_request)

@pytest.mark.asyncio
@respx.mock
async def test_groq_network_error(groq_provider, basic_request):
    respx.post().mock(side_effect=httpx.NetworkError("Network issue"))
    with pytest.raises(ProviderNetworkError):
        await groq_provider.send_chat_completion(basic_request)

@pytest.mark.asyncio
async def test_groq_missing_api_key(basic_request):
    provider = GroqProvider()
    provider.api_key = None
    with pytest.raises(ProviderAuthenticationError):
        await provider.send_chat_completion(basic_request)

@pytest.mark.asyncio
async def test_groq_unsupported_model(groq_provider):
    req = ChatCompletionRequest(model="gpt-4", messages=[ChatMessage(role="user", content="Hi")])
    with pytest.raises(ProviderUnsupportedModelError):
        await groq_provider.send_chat_completion(req)
