import asyncio
import time
from enum import Enum
from typing import Dict, List, Optional
from dataclasses import dataclass

from app.providers.config_registry import configurable_registry
from app.models.provider_config import HealthCheckSettings
from app.core.logging import get_logger
from app.core.error_classifier import classify_error
from app.providers.errors import ProviderError

logger = get_logger(__name__)

class ProviderHealthState(str, Enum):
    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    RECOVERING = "recovering"

@dataclass
class ProviderHealthStatus:
    provider_name: str
    state: ProviderHealthState = ProviderHealthState.UNKNOWN
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_check_time: float = 0.0
    last_latency: Optional[float] = None
    last_error_category: Optional[str] = None

class ProviderHealthManager:
    """Centralized provider health manager."""
    
    def __init__(self, registry=None):
        self.registry = registry or configurable_registry
        self.health_states: Dict[str, ProviderHealthStatus] = {}
        self._tasks: Dict[str, asyncio.Task] = {}
        self._running = False
        
    def get_status(self, provider_name: str) -> ProviderHealthStatus:
        """Get the current health status of a provider."""
        if provider_name not in self.health_states:
            self.health_states[provider_name] = ProviderHealthStatus(provider_name=provider_name)
        return self.health_states[provider_name]

    def get_available_providers(self, provider_names: List[str]) -> List[str]:
        """Filter the list of providers, returning only those not marked as UNHEALTHY."""
        available = []
        for name in provider_names:
            status = self.get_status(name)
            if status.state != ProviderHealthState.UNHEALTHY:
                available.append(name)
        return available

    async def start(self):
        """Start the background health check tasks."""
        if self._running:
            return
        self._running = True
        
        # We start a task for each registered provider
        for provider_name in self.registry.list_providers():
            config = self.registry.get_provider_config(provider_name)
            if config and config.health_check_settings.enabled:
                self._tasks[provider_name] = asyncio.create_task(self._health_check_loop(provider_name))

    async def stop(self):
        """Stop all background health check tasks."""
        self._running = False
        for task in self._tasks.values():
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        self._tasks.clear()

    async def _health_check_loop(self, provider_name: str):
        """Background loop to periodically check a provider's health."""
        while self._running:
            try:
                config = self.registry.get_provider_config(provider_name)
                # If config is removed or health check disabled, stop this loop
                if not config or not config.health_check_settings.enabled:
                    break
                
                settings = config.health_check_settings
                status = self.get_status(provider_name)
                
                now = time.time()
                time_since_last_check = now - status.last_check_time
                
                # Only check if interval has passed
                if time_since_last_check >= settings.interval_seconds or status.last_check_time == 0.0:
                    await self._perform_check(provider_name, settings)
                
                # Re-fetch after check, compute sleep time
                status = self.get_status(provider_name)
                sleep_time = max(0.1, settings.interval_seconds - (time.time() - status.last_check_time))
                await asyncio.sleep(sleep_time)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop for {provider_name}: {e}")
                await asyncio.sleep(5)

    async def _perform_check(self, provider_name: str, settings: HealthCheckSettings):
        """Execute a single health check for the given provider."""
        status = self.get_status(provider_name)
        
        try:
            provider = self.registry.get_provider(provider_name)
        except KeyError:
            return  # Provider no longer in registry
            
        start_time = time.time()
        is_healthy = False
        error_category = None
        
        try:
            is_healthy = await asyncio.wait_for(provider.check_health(), timeout=settings.timeout_seconds)
        except asyncio.TimeoutError:
            error_category = "timeout_error"
            is_healthy = False
        except ProviderError as e:
            error_category = e.error_type.value
            is_healthy = False
        except Exception as e:
            classification = classify_error(provider_name, original_error=str(e))
            error_category = classification.error_type.value
            is_healthy = False
            
        latency = time.time() - start_time
        status.last_check_time = time.time()
        
        old_state = status.state
        
        if is_healthy:
            status.last_latency = latency
            status.consecutive_successes += 1
            status.consecutive_failures = 0
            status.last_error_category = None
            
            if status.state == ProviderHealthState.UNHEALTHY or status.state == ProviderHealthState.RECOVERING:
                status.state = ProviderHealthState.RECOVERING
                if status.consecutive_successes >= settings.healthy_threshold:
                    status.state = ProviderHealthState.HEALTHY
            else:
                status.state = ProviderHealthState.HEALTHY
                
        else:
            status.last_error_category = error_category
            status.consecutive_failures += 1
            status.consecutive_successes = 0
            
            if status.state == ProviderHealthState.HEALTHY or status.state == ProviderHealthState.DEGRADED:
                if status.consecutive_failures >= settings.unhealthy_threshold:
                    status.state = ProviderHealthState.UNHEALTHY
                else:
                    status.state = ProviderHealthState.DEGRADED
            elif status.state == ProviderHealthState.UNKNOWN:
                if status.consecutive_failures >= settings.unhealthy_threshold:
                    status.state = ProviderHealthState.UNHEALTHY
                else:
                    status.state = ProviderHealthState.DEGRADED
            elif status.state == ProviderHealthState.RECOVERING:
                # If we were recovering but get a failure, back to UNHEALTHY
                status.state = ProviderHealthState.UNHEALTHY
                
        if old_state != status.state:
            logger.info(
                f"Provider {provider_name} health state transitioned from {old_state.value} to {status.state.value}",
                extra={
                    "provider": provider_name,
                    "old_state": old_state.value,
                    "new_state": status.state.value,
                    "consecutive_failures": status.consecutive_failures,
                    "consecutive_successes": status.consecutive_successes,
                    "latency": latency,
                    "error_category": error_category
                }
            )

# Global singleton
health_manager = ProviderHealthManager()
