import pytest
import asyncio
import os
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock

from app.providers.config_registry import (
    ConfigurableProviderRegistry,
    ConfigLoadError,
    ConfigValidationError
)
from app.models.provider_config import ProviderConfig, ProviderRegistryConfig
from app.models.chat import ChatCompletionRequest, ChatMessage


class TestProviderConfig:
    """Test provider configuration model validation."""
    
    def test_valid_provider_config(self):
        """Test creating a valid provider configuration."""
        config = ProviderConfig(
            name="test-provider",
            adapter_type="app.providers.mock.MockProvider",
            enabled=True,
            supported_models=["model-1", "model-2"],
            priority=100,
            timeout_seconds=30.0
        )
        assert config.name == "test-provider"
        assert config.enabled is True
        assert len(config.supported_models) == 2
    
    def test_invalid_provider_name(self):
        """Test that invalid provider names are rejected."""
        with pytest.raises(ValueError, match="must contain only alphanumeric"):
            ProviderConfig(
                name="invalid@name!",
                adapter_type="app.providers.mock.MockProvider",
                enabled=True,
                supported_models=["model-1"]
            )
    
    def test_invalid_adapter_type(self):
        """Test that invalid adapter types are rejected."""
        with pytest.raises(ValueError, match="must be a valid module path"):
            ProviderConfig(
                name="test",
                adapter_type="invalid_module",
                enabled=True,
                supported_models=["model-1"]
            )
    
    def test_empty_supported_models(self):
        """Test that empty supported models list is rejected."""
        with pytest.raises(ValueError, match="at least 1 item"):
            ProviderConfig(
                name="test",
                adapter_type="app.providers.mock.MockProvider",
                enabled=True,
                supported_models=[]
            )
    
    def test_duplicate_provider_names(self):
        """Test that duplicate provider names are rejected."""
        with pytest.raises(ValueError, match="Provider names must be unique"):
            ProviderRegistryConfig(
                providers=[
                    ProviderConfig(
                        name="duplicate",
                        adapter_type="app.providers.mock.MockProvider",
                        enabled=True,
                        supported_models=["model-1"]
                    ),
                    ProviderConfig(
                        name="duplicate",
                        adapter_type="app.providers.mock.MockProvider",
                        enabled=True,
                        supported_models=["model-2"]
                    )
                ]
            )
    
    def test_retry_policy_validation(self):
        """Test retry policy field validation."""
        with pytest.raises(ValueError, match="less than or equal to 10"):
            ProviderConfig(
                name="test",
                adapter_type="app.providers.mock.MockProvider",
                enabled=True,
                supported_models=["model-1"],
                retry_policy={"max_attempts": 15}  # type: ignore
            )


class TestConfigurableProviderRegistry:
    """Test configurable provider registry functionality."""
    
    @pytest.fixture
    def valid_config_path(self):
        """Create a temporary valid config file."""
        config_data = {
            "providers": [
                {
                    "name": "mock",
                    "adapter_type": "app.providers.mock.MockProvider",
                    "enabled": True,
                    "supported_models": ["mock-gpt-4", "mock-gpt-3.5-turbo"],
                    "priority": 100,
                    "timeout_seconds": 30.0,
                    "retry_policy": {
                        "max_attempts": 3,
                        "backoff_factor": 2.0,
                        "initial_delay": 1.0
                    },
                    "health_check_settings": {
                        "enabled": True,
                        "interval_seconds": 60,
                        "timeout_seconds": 5.0,
                        "unhealthy_threshold": 3,
                        "healthy_threshold": 1
                    },
                    "routing_weight": 1.0,
                    "capabilities": {
                        "streaming": False,
                        "function_calling": False,
                        "vision": False,
                        "parallel_requests": True
                    },
                    "env_var_prefix": None
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        os.unlink(temp_path)
    
    @pytest.fixture
    def invalid_config_path(self):
        """Create a temporary invalid config file."""
        config_data = {
            "providers": [
                {
                    "name": "invalid@name!",
                    "adapter_type": "invalid_module",
                    "enabled": True,
                    "supported_models": [],  # Empty list
                    "priority": 100,
                    "timeout_seconds": 30.0,
                    "retry_policy": {
                        "max_attempts": 15  # Exceeds max
                    }
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        os.unlink(temp_path)
    
    def test_load_valid_config(self, valid_config_path):
        """Test loading a valid configuration file."""
        registry = ConfigurableProviderRegistry(config_path=valid_config_path)
        config = registry.load_config()
        
        assert isinstance(config, ProviderRegistryConfig)
        assert len(config.providers) == 1
        assert config.providers[0].name == "mock"
    
    def test_load_invalid_config(self, invalid_config_path):
        """Test that invalid configuration is rejected."""
        registry = ConfigurableProviderRegistry(config_path=invalid_config_path)
        
        with pytest.raises(ConfigValidationError):
            registry.load_config()
    
    def test_load_nonexistent_config(self):
        """Test loading a non-existent config file."""
        registry = ConfigurableProviderRegistry(config_path="/nonexistent/path/config.yaml")
        
        with pytest.raises(ConfigLoadError, match="Config file not found"):
            registry.load_config()
    
    def test_initialize_from_config(self, valid_config_path):
        """Test initializing registry from config file."""
        registry = ConfigurableProviderRegistry(config_path=valid_config_path)
        registry.initialize_from_config()
        
        providers = registry.list_providers()
        assert "mock" in providers
        assert len(providers) == 1
    
    def test_provider_instance_creation(self, valid_config_path):
        """Test that provider instances are created correctly."""
        registry = ConfigurableProviderRegistry(config_path=valid_config_path)
        registry.initialize_from_config()
        
        provider = registry.get_provider("mock")
        assert provider is not None
        assert hasattr(provider, 'get_supported_models')
    
    def test_get_provider_config(self, valid_config_path):
        """Test retrieving provider configuration."""
        registry = ConfigurableProviderRegistry(config_path=valid_config_path)
        registry.initialize_from_config()
        
        config = registry.get_provider_config("mock")
        assert config is not None
        assert config.name == "mock"
        assert config.priority == 100
    
    def test_get_provider_by_priority(self, valid_config_path):
        """Test getting providers sorted by priority."""
        # Create config with multiple providers
        config_data = {
            "providers": [
                {
                    "name": "low-priority",
                    "adapter_type": "app.providers.mock.MockProvider",
                    "enabled": True,
                    "supported_models": ["model-1"],
                    "priority": 10,
                    "timeout_seconds": 30.0
                },
                {
                    "name": "high-priority",
                    "adapter_type": "app.providers.mock.MockProvider",
                    "enabled": True,
                    "supported_models": ["model-2"],
                    "priority": 100,
                    "timeout_seconds": 30.0
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            registry = ConfigurableProviderRegistry(config_path=temp_path)
            registry.initialize_from_config()
            
            providers_by_priority = registry.get_provider_by_priority()
            assert providers_by_priority == ["high-priority", "low-priority"]
        finally:
            os.unlink(temp_path)
    
    def test_get_providers_by_capability(self, valid_config_path):
        """Test filtering providers by capability."""
        # Create config with different capabilities
        config_data = {
            "providers": [
                {
                    "name": "streaming-provider",
                    "adapter_type": "app.providers.mock.MockProvider",
                    "enabled": True,
                    "supported_models": ["model-1"],
                    "priority": 100,
                    "timeout_seconds": 30.0,
                    "capabilities": {
                        "streaming": True,
                        "function_calling": False,
                        "vision": False,
                        "parallel_requests": True
                    }
                },
                {
                    "name": "non-streaming-provider",
                    "adapter_type": "app.providers.mock.MockProvider",
                    "enabled": True,
                    "supported_models": ["model-2"],
                    "priority": 90,
                    "timeout_seconds": 30.0,
                    "capabilities": {
                        "streaming": False,
                        "function_calling": False,
                        "vision": False,
                        "parallel_requests": True
                    }
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            registry = ConfigurableProviderRegistry(config_path=temp_path)
            registry.initialize_from_config()
            
            streaming_providers = registry.get_providers_by_capability("streaming")
            assert "streaming-provider" in streaming_providers
            assert "non-streaming-provider" not in streaming_providers
        finally:
            os.unlink(temp_path)
    
    def test_enabled_disabled_providers(self):
        """Test that disabled providers are not registered."""
        config_data = {
            "providers": [
                {
                    "name": "enabled-provider",
                    "adapter_type": "app.providers.mock.MockProvider",
                    "enabled": True,
                    "supported_models": ["model-1"],
                    "priority": 100,
                    "timeout_seconds": 30.0
                },
                {
                    "name": "disabled-provider",
                    "adapter_type": "app.providers.mock.MockProvider",
                    "enabled": False,
                    "supported_models": ["model-2"],
                    "priority": 90,
                    "timeout_seconds": 30.0
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            registry = ConfigurableProviderRegistry(config_path=temp_path)
            registry.initialize_from_config()
            
            providers = registry.list_providers()
            assert "enabled-provider" in providers
            assert "disabled-provider" not in providers
            assert len(providers) == 1
        finally:
            os.unlink(temp_path)


class TestHotReload:
    """Test hot-reload functionality."""
    
    @pytest.fixture
    def config_with_reload(self):
        """Create a config file that can be modified for hot-reload testing."""
        config_data = {
            "providers": [
                {
                    "name": "mock",
                    "adapter_type": "app.providers.mock.MockProvider",
                    "enabled": True,
                    "supported_models": ["mock-gpt-4"],
                    "priority": 100,
                    "timeout_seconds": 30.0
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        os.unlink(temp_path)
    
    @pytest.mark.asyncio
    async def test_hot_reload_initialization(self, config_with_reload):
        """Test starting hot-reload watcher."""
        registry = ConfigurableProviderRegistry(config_path=config_with_reload)
        registry.initialize_from_config()
        
        await registry.start_hot_reload()
        
        # Give it a moment to start
        await asyncio.sleep(0.1)
        
        assert registry._reload_task is not None
        assert not registry._reload_task.done()
        
        await registry.stop_hot_reload()
    
    @pytest.mark.asyncio
    async def test_hot_reload_stop(self, config_with_reload):
        """Test stopping hot-reload watcher."""
        registry = ConfigurableProviderRegistry(config_path=config_with_reload)
        registry.initialize_from_config()
        
        await registry.start_hot_reload()
        await registry.stop_hot_reload()
        
        assert registry._reload_task is None
    
    @pytest.mark.asyncio
    async def test_active_request_tracking(self, config_with_reload):
        """Test active request tracking for hot-reload safety."""
        registry = ConfigurableProviderRegistry(config_path=config_with_reload)
        registry.initialize_from_config()
        
        assert registry.get_active_request_count() == 0
        
        await registry.increment_active_requests()
        assert registry.get_active_request_count() == 1
        
        await registry.increment_active_requests()
        assert registry.get_active_request_count() == 2
        
        await registry.decrement_active_requests()
        assert registry.get_active_request_count() == 1
        
        await registry.decrement_active_requests()
        assert registry.get_active_request_count() == 0
    
    @pytest.mark.asyncio
    async def test_safe_reload_with_active_requests(self, config_with_reload):
        """Test that reload waits for active requests to complete."""
        registry = ConfigurableProviderRegistry(config_path=config_with_reload)
        registry.initialize_from_config()
        
        # Simulate active request
        await registry.increment_active_requests()
        
        # Try to reload - should be delayed
        reload_task = asyncio.create_task(registry._safe_reload())
        
        # Give it time to check active requests
        await asyncio.sleep(0.1)
        
        # Complete the active request
        await registry.decrement_active_requests()
        
        # Reload should complete
        try:
            await asyncio.wait_for(reload_task, timeout=2.0)
        except asyncio.TimeoutError:
            pytest.fail("Reload did not complete after active requests finished")
    
    @pytest.mark.asyncio
    async def test_reload_rollback_on_validation_error(self, config_with_reload):
        """Test that invalid reload rolls back to previous config."""
        registry = ConfigurableProviderRegistry(config_path=config_with_reload)
        registry.initialize_from_config()
        
        original_providers = registry.list_providers().copy()
        
        # Modify config to be invalid
        invalid_config = {
            "providers": [
                {
                    "name": "invalid@name!",
                    "adapter_type": "invalid_module",
                    "enabled": True,
                    "supported_models": [],
                    "priority": 100,
                    "timeout_seconds": 30.0
                }
            ]
        }
        
        with open(config_with_reload, 'w') as f:
            yaml.dump(invalid_config, f)
        
        # Attempt reload - should fail and rollback
        with pytest.raises(ConfigValidationError):
            await registry._safe_reload()
        
        # Verify rollback
        assert registry.list_providers() == original_providers


class TestRouterIntegration:
    """Test router integration with configurable registry."""
    
    @pytest.fixture
    def registry_with_providers(self):
        """Create a registry with test providers."""
        config_data = {
            "providers": [
                {
                    "name": "mock-primary",
                    "adapter_type": "app.providers.mock.MockProvider",
                    "enabled": True,
                    "supported_models": ["mock-gpt-4", "mock-gpt-3.5-turbo"],
                    "priority": 100,
                    "timeout_seconds": 30.0
                },
                {
                    "name": "mock-secondary",
                    "adapter_type": "app.providers.mock.MockProvider",
                    "enabled": True,
                    "supported_models": ["mock-claude"],
                    "priority": 50,
                    "timeout_seconds": 30.0
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        registry = ConfigurableProviderRegistry(config_path=temp_path)
        registry.initialize_from_config()
        
        yield registry
        
        os.unlink(temp_path)
    
    @pytest.mark.asyncio
    async def test_router_uses_registry(self, registry_with_providers):
        """Test that router consumes registry correctly."""
        from app.services.router import RoutingEngine
        
        router = RoutingEngine(registry=registry_with_providers)
        
        request = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="test")],
            model="mock-gpt-4"
        )
        
        response = await router.route_chat_completion(request, target_provider="mock-primary")
        
        assert response is not None
        assert response.provider == "mock"
    
    @pytest.mark.asyncio
    async def test_router_priority_selection(self, registry_with_providers):
        """Test that router selects provider by priority when none specified."""
        from app.services.router import RoutingEngine
        
        router = RoutingEngine(registry=registry_with_providers)
        
        request = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="test")],
            model="mock-gpt-4"
        )
        
        # Should select mock-primary (higher priority)
        response = await router.route_chat_completion(request)
        
        assert response is not None
    
    def test_router_get_available_providers(self, registry_with_providers):
        """Test router can list available providers."""
        from app.services.router import RoutingEngine
        
        router = RoutingEngine(registry=registry_with_providers)
        
        providers = router.get_available_providers()
        assert "mock-primary" in providers
        assert "mock-secondary" in providers
    
    def test_router_get_provider_config(self, registry_with_providers):
        """Test router can get provider configuration."""
        from app.services.router import RoutingEngine
        
        router = RoutingEngine(registry=registry_with_providers)
        
        config = router.get_provider_config("mock-primary")
        assert config is not None
        assert config.name == "mock-primary"
        assert config.priority == 100
    
    def test_router_get_providers_by_capability(self, registry_with_providers):
        """Test router can filter providers by capability."""
        from app.services.router import RoutingEngine
        
        router = RoutingEngine(registry=registry_with_providers)
        
        # Mock providers have parallel_requests=True by default
        providers = router.get_providers_by_capability("parallel_requests")
        assert len(providers) >= 1