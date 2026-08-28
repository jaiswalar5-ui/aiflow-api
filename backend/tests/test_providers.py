import pytest
from app.providers.registry import ProviderRegistry
from app.providers.config_registry import ConfigurableProviderRegistry
from app.providers.mock import MockProvider, MockFailingProvider
from app.providers.errors import ProviderRateLimitError, ProviderUnsupportedModelError
from app.models.chat import ChatCompletionRequest, ChatMessage
from app.services.router import RoutingEngine

@pytest.fixture
def empty_registry():
    return ProviderRegistry()

@pytest.fixture
def empty_config_registry():
    return ConfigurableProviderRegistry()

@pytest.fixture
def mock_request():
    return ChatCompletionRequest(
        model="mock-gpt",
        messages=[ChatMessage(role="user", content="Hello")]
    )

def test_provider_registry(empty_registry):
    mock = MockProvider()
    empty_registry.register_provider("mock", mock)
    
    assert empty_registry.list_providers() == ["mock"]
    assert empty_registry.get_provider("mock") == mock
    
    with pytest.raises(KeyError):
        empty_registry.get_provider("nonexistent")

def test_config_registry(empty_config_registry):
    mock = MockProvider()
    empty_config_registry.register_provider("mock", mock)
    
    assert empty_config_registry.list_providers() == ["mock"]
    assert empty_config_registry.get_provider("mock") == mock
    
    with pytest.raises(KeyError):
        empty_config_registry.get_provider("nonexistent")

@pytest.mark.asyncio
async def test_mock_provider_success(mock_request):
    provider = MockProvider()
    assert await provider.check_health() is True
    
    response = await provider.send_chat_completion(mock_request)
    assert response.choices[0].message.content == "This is a mock provider response."
    assert response.model == "mock-gpt"

@pytest.mark.asyncio
async def test_mock_failing_provider(mock_request):
    provider = MockFailingProvider(ProviderRateLimitError("Rate limit exceeded"))
    assert await provider.check_health() is False
    
    with pytest.raises(ProviderRateLimitError):
        await provider.send_chat_completion(mock_request)

@pytest.mark.asyncio
async def test_routing_engine_success(mock_request, empty_config_registry):
    # Register with configurable registry
    empty_config_registry.register_provider("success_mock", MockProvider())
    
    engine = RoutingEngine(registry=empty_config_registry)
    response = await engine.route_chat_completion(mock_request, target_provider="success_mock")
    
    assert response.choices[0].message.content == "This is a mock provider response."

@pytest.mark.asyncio
async def test_routing_engine_unsupported_model(empty_config_registry):
    empty_config_registry.register_provider("mock", MockProvider())
    
    engine = RoutingEngine(registry=empty_config_registry)
    req = ChatCompletionRequest(model="unsupported-model", messages=[ChatMessage(role="user", content="Hi")])
    
    with pytest.raises(ProviderUnsupportedModelError):
        await engine.route_chat_completion(req, target_provider="mock")
