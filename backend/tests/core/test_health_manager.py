import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from app.core.health_manager import ProviderHealthManager, ProviderHealthState
from app.models.provider_config import ProviderConfig, HealthCheckSettings
from app.providers.errors import ProviderError, ErrorType, ServerError

@pytest.fixture
def mock_registry():
    registry = MagicMock()
    return registry

@pytest.fixture
def health_manager(mock_registry):
    return ProviderHealthManager(registry=mock_registry)

@pytest.mark.asyncio
async def test_health_check_success(health_manager, mock_registry):
    # Setup
    config = ProviderConfig(
        name="test_provider",
        adapter_type="app.providers.mock.MockProvider",
        supported_models=["model"],
        health_check_settings=HealthCheckSettings(
            enabled=True, interval_seconds=10, timeout_seconds=1.0,
            unhealthy_threshold=2, healthy_threshold=2
        )
    )
    mock_registry.get_provider_config.return_value = config
    
    mock_provider = AsyncMock()
    mock_provider.check_health.return_value = True
    mock_registry.get_provider.return_value = mock_provider
    
    # Run one check
    await health_manager._perform_check("test_provider", config.health_check_settings)
    
    status = health_manager.get_status("test_provider")
    assert status.state == ProviderHealthState.HEALTHY
    assert status.consecutive_successes == 1
    assert status.last_latency is not None

@pytest.mark.asyncio
async def test_health_check_failure_transitions(health_manager, mock_registry):
    config = ProviderConfig(
        name="test_provider",
        adapter_type="app.providers.mock.MockProvider",
        supported_models=["model"],
        health_check_settings=HealthCheckSettings(
            enabled=True, interval_seconds=10, timeout_seconds=1.0,
            unhealthy_threshold=3, healthy_threshold=2
        )
    )
    mock_registry.get_provider_config.return_value = config
    
    mock_provider = AsyncMock()
    # Raise a provider error
    mock_provider.check_health.side_effect = ServerError("fail")
    mock_registry.get_provider.return_value = mock_provider
    
    # 1st failure -> DEGRADED
    await health_manager._perform_check("test_provider", config.health_check_settings)
    status = health_manager.get_status("test_provider")
    assert status.state == ProviderHealthState.DEGRADED
    assert status.consecutive_failures == 1
    assert status.last_error_category == "server_error"
    
    # 2nd failure -> DEGRADED
    await health_manager._perform_check("test_provider", config.health_check_settings)
    status = health_manager.get_status("test_provider")
    assert status.state == ProviderHealthState.DEGRADED
    assert status.consecutive_failures == 2
    
    # 3rd failure -> UNHEALTHY
    await health_manager._perform_check("test_provider", config.health_check_settings)
    status = health_manager.get_status("test_provider")
    assert status.state == ProviderHealthState.UNHEALTHY
    assert status.consecutive_failures == 3

@pytest.mark.asyncio
async def test_health_check_recovery_transitions(health_manager, mock_registry):
    config = ProviderConfig(
        name="test_provider",
        adapter_type="app.providers.mock.MockProvider",
        supported_models=["model"],
        health_check_settings=HealthCheckSettings(
            enabled=True, interval_seconds=10, timeout_seconds=1.0,
            unhealthy_threshold=2, healthy_threshold=2
        )
    )
    mock_registry.get_provider_config.return_value = config
    
    # Setup as UNHEALTHY
    status = health_manager.get_status("test_provider")
    status.state = ProviderHealthState.UNHEALTHY
    status.consecutive_failures = 2
    
    mock_provider = AsyncMock()
    mock_provider.check_health.return_value = True
    mock_registry.get_provider.return_value = mock_provider
    
    # 1st success -> RECOVERING
    await health_manager._perform_check("test_provider", config.health_check_settings)
    assert status.state == ProviderHealthState.RECOVERING
    assert status.consecutive_successes == 1
    
    # 2nd success -> HEALTHY
    await health_manager._perform_check("test_provider", config.health_check_settings)
    assert status.state == ProviderHealthState.HEALTHY
    assert status.consecutive_successes == 2

@pytest.mark.asyncio
async def test_health_check_timeout(health_manager, mock_registry):
    config = ProviderConfig(
        name="test_provider",
        adapter_type="app.providers.mock.MockProvider",
        supported_models=["model"],
        health_check_settings=HealthCheckSettings(
            enabled=True, interval_seconds=10, timeout_seconds=1.0,
            unhealthy_threshold=1, healthy_threshold=1
        )
    )
    mock_registry.get_provider_config.return_value = config
    
    mock_provider = AsyncMock()
    async def slow_health():
        await asyncio.sleep(1.5)
        return True
    mock_provider.check_health.side_effect = slow_health
    mock_registry.get_provider.return_value = mock_provider
    
    await health_manager._perform_check("test_provider", config.health_check_settings)
    status = health_manager.get_status("test_provider")
    assert status.state == ProviderHealthState.UNHEALTHY
    assert status.last_error_category == "timeout_error"

@pytest.mark.asyncio
async def test_get_available_providers(health_manager):
    # Setup statuses
    health_manager.get_status("p1").state = ProviderHealthState.HEALTHY
    health_manager.get_status("p2").state = ProviderHealthState.DEGRADED
    health_manager.get_status("p3").state = ProviderHealthState.UNHEALTHY
    health_manager.get_status("p4").state = ProviderHealthState.RECOVERING
    health_manager.get_status("p5").state = ProviderHealthState.UNKNOWN
    
    available = health_manager.get_available_providers(["p1", "p2", "p3", "p4", "p5"])
    assert "p3" not in available
    assert "p1" in available
    assert "p2" in available
    assert "p4" in available
    assert "p5" in available

@pytest.mark.asyncio
async def test_passive_health_success(health_manager, mock_registry):
    config = ProviderConfig(
        name="test_provider",
        adapter_type="app.providers.mock.MockProvider",
        supported_models=["model"],
        health_check_settings=HealthCheckSettings(
            enabled=True, healthy_threshold=2, unhealthy_threshold=2
        )
    )
    mock_registry.get_provider_config.return_value = config
    
    # 1st success
    await health_manager.report_success("test_provider", latency=0.1)
    status = health_manager.get_status("test_provider")
    assert status.state == ProviderHealthState.HEALTHY
    assert status.consecutive_successes == 1
    assert status.last_latency == 0.1
    assert status.last_success_time > 0
    
    # Force to unhealthy to test recovery
    status.state = ProviderHealthState.UNHEALTHY
    status.consecutive_successes = 0
    
    # 1st success in recovery -> RECOVERING
    await health_manager.report_success("test_provider")
    assert status.state == ProviderHealthState.RECOVERING
    assert status.consecutive_successes == 1
    
    # 2nd success in recovery -> HEALTHY
    await health_manager.report_success("test_provider")
    assert status.state == ProviderHealthState.HEALTHY
    assert status.consecutive_successes == 2

@pytest.mark.asyncio
async def test_passive_health_failure_transitions(health_manager, mock_registry):
    config = ProviderConfig(
        name="test_provider",
        adapter_type="app.providers.mock.MockProvider",
        supported_models=["model"],
        health_check_settings=HealthCheckSettings(
            enabled=True, healthy_threshold=2, unhealthy_threshold=2
        )
    )
    mock_registry.get_provider_config.return_value = config
    
    # 1st failure -> DEGRADED
    await health_manager.report_failure("test_provider", ErrorType.SERVER_ERROR)
    status = health_manager.get_status("test_provider")
    assert status.state == ProviderHealthState.DEGRADED
    assert status.consecutive_failures == 1
    assert status.last_error_category == "server_error"
    assert status.last_failure_time > 0
    assert len(status.error_history) == 1
    assert status.error_history[0]["error"] == "server_error"
    
    # 2nd failure -> UNHEALTHY
    await health_manager.report_failure("test_provider", ErrorType.RATE_LIMIT_ERROR)
    assert status.state == ProviderHealthState.UNHEALTHY
    assert status.consecutive_failures == 2
    assert status.last_error_category == "rate_limit_error"
    assert len(status.error_history) == 2

@pytest.mark.asyncio
async def test_passive_health_ignores_client_errors(health_manager, mock_registry):
    config = ProviderConfig(
        name="test_provider",
        adapter_type="app.providers.mock.MockProvider",
        supported_models=["model"],
        health_check_settings=HealthCheckSettings(
            enabled=True, healthy_threshold=2, unhealthy_threshold=2
        )
    )
    mock_registry.get_provider_config.return_value = config
    
    # Auth error should be ignored
    await health_manager.report_failure("test_provider", ErrorType.AUTHENTICATION_ERROR)
    status = health_manager.get_status("test_provider")
    assert status.state == ProviderHealthState.UNKNOWN
    assert status.consecutive_failures == 0
    assert len(status.error_history) == 0

    # Invalid request should be ignored
    await health_manager.report_failure("test_provider", ErrorType.INVALID_REQUEST)
    assert status.state == ProviderHealthState.UNKNOWN
    assert status.consecutive_failures == 0

@pytest.mark.asyncio
async def test_passive_health_history_bounded(health_manager, mock_registry):
    config = ProviderConfig(
        name="test_provider",
        adapter_type="app.providers.mock.MockProvider",
        supported_models=["model"],
        health_check_settings=HealthCheckSettings(
            enabled=True, healthy_threshold=2, unhealthy_threshold=10
        )
    )
    mock_registry.get_provider_config.return_value = config
    
    for _ in range(15):
        await health_manager.report_failure("test_provider", ErrorType.NETWORK_ERROR)
        
    status = health_manager.get_status("test_provider")
    assert status.consecutive_failures == 15
    assert len(status.error_history) <= 10  # Max length is 10

@pytest.mark.asyncio
async def test_passive_health_concurrent_updates(health_manager, mock_registry):
    config = ProviderConfig(
        name="test_provider",
        adapter_type="app.providers.mock.MockProvider",
        supported_models=["model"],
        health_check_settings=HealthCheckSettings(
            enabled=True, healthy_threshold=2, unhealthy_threshold=10
        )
    )
    mock_registry.get_provider_config.return_value = config
    
    # Simulate many concurrent failures
    import asyncio
    tasks = []
    for _ in range(50):
        tasks.append(health_manager.report_failure("test_provider", ErrorType.SERVER_ERROR))
        
    await asyncio.gather(*tasks)
    
    status = health_manager.get_status("test_provider")
    assert status.consecutive_failures == 50
    assert len(status.error_history) == 10  # Max size
