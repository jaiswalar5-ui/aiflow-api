import asyncio
import time
from collections import deque
from enum import Enum
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from app.providers.config_registry import configurable_registry
from app.models.provider_config import HealthCheckSettings
from app.core.logging import get_logger
from app.core.error_classifier import classify_error
from app.providers.errors import ProviderError, ErrorType

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
    last_success_time: float = 0.0
    last_failure_time: float = 0.0
    error_history: deque = field(default_factory=lambda: deque(maxlen=10))

class ProviderHealthManager:
    """Centralized provider health manager."""
    
    def __init__(self, registry=None):
        self.registry = registry or configurable_registry
        self.health_states: Dict[str, ProviderHealthStatus] = {}
        self.health_locks: Dict[str, asyncio.Lock] = {}
        self._tasks: Dict[str, asyncio.Task] = {}
        self._running = False
        
    def get_status(self, provider_name: str) -> ProviderHealthStatus:
        """Get the current health status of a provider."""
        if provider_name not in self.health_states:
            self.health_states[provider_name] = ProviderHealthStatus(provider_name=provider_name)
        return self.health_states[provider_name]

    def _get_lock(self, provider_name: str) -> asyncio.Lock:
        if provider_name not in self.health_locks:
            self.health_locks[provider_name] = asyncio.Lock()
        return self.health_locks[provider_name]

    def get_available_providers(self, provider_names: List[str]) -> List[str]:
        """Filter the list of providers, returning only those not marked as UNHEALTHY."""
        available = []
        for name in provider_names:
            status = self.get_status(name)
            if status.state != ProviderHealthState.UNHEALTHY:
                available.append(name)
        return available

    def get_snapshot(self, provider_name: str) -> dict:
        """Return a safe, bounded snapshot of the provider's health state."""
        from dataclasses import asdict
        status = self.get_status(provider_name)
        snapshot = asdict(status)
        snapshot["error_history"] = list(snapshot["error_history"])
        return snapshot

    async def report_success(self, provider_name: str, latency: Optional[float] = None):
        """Passively report a successful request."""
        config = self.registry.get_provider_config(provider_name)
        if not config or not config.health_check_settings.enabled:
            return
            
        settings = config.health_check_settings
        
        async with self._get_lock(provider_name):
            status = self.get_status(provider_name)
            status.last_success_time = time.time()
            if latency is not None:
                status.last_latency = latency
                
            status.consecutive_successes = min(10000, status.consecutive_successes + 1)
            status.consecutive_failures = 0
            status.last_error_category = None
            
            old_state = status.state
            
            if status.state == ProviderHealthState.UNHEALTHY or status.state == ProviderHealthState.RECOVERING:
                status.state = ProviderHealthState.RECOVERING
                if status.consecutive_successes >= settings.healthy_threshold:
                    status.state = ProviderHealthState.HEALTHY
            elif status.state == ProviderHealthState.DEGRADED or status.state == ProviderHealthState.UNKNOWN:
                status.state = ProviderHealthState.HEALTHY
                
            self._log_transition(provider_name, status, old_state, "passive_success")

    async def report_failure(self, provider_name: str, error_type: ErrorType):
        """Passively report a failed request."""
        config = self.registry.get_provider_config(provider_name)
        if not config or not config.health_check_settings.enabled:
            return
            
        # Ignore client-side invalid requests and auth/config errors
        if error_type in (ErrorType.INVALID_REQUEST, ErrorType.AUTHENTICATION_ERROR, ErrorType.UNSUPPORTED_MODEL_ERROR):
            return
            
        settings = config.health_check_settings
        
        async with self._get_lock(provider_name):
            status = self.get_status(provider_name)
            status.last_failure_time = time.time()
            status.last_error_category = error_type.value
            status.error_history.append({"time": time.time(), "error": error_type.value})
            
            status.consecutive_failures = min(10000, status.consecutive_failures + 1)
            status.consecutive_successes = 0
            
            old_state = status.state
            
            if status.state in (ProviderHealthState.HEALTHY, ProviderHealthState.DEGRADED, ProviderHealthState.UNKNOWN):
                if status.consecutive_failures >= settings.unhealthy_threshold:
                    status.state = ProviderHealthState.UNHEALTHY
                else:
                    status.state = ProviderHealthState.DEGRADED
            elif status.state == ProviderHealthState.RECOVERING:
                status.state = ProviderHealthState.UNHEALTHY
                
            self._log_transition(provider_name, status, old_state, "passive_failure")

    def _log_transition(self, provider_name: str, status: ProviderHealthStatus, old_state: ProviderHealthState, source: str):
        if old_state != status.state:
            logger.info(
                f"Provider {provider_name} health state transitioned from {old_state.value} to {status.state.value}",
                extra={
                    "provider": provider_name,
                    "old_state": old_state.value,
                    "new_state": status.state.value,
                    "consecutive_failures": status.consecutive_failures,
                    "consecutive_successes": status.consecutive_successes,
                    "latency": status.last_latency,
                    "error_category": status.last_error_category,
                    "source": source
                }
            )

    async def start(self):
        """Start the background health check tasks."""
        if self._running:
            return
        self._running = True
        
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
                if not config or not config.health_check_settings.enabled:
                    break
                
                settings = config.health_check_settings
                status = self.get_status(provider_name)
                
                now = time.time()
                time_since_last_check = now - status.last_check_time
                
                if time_since_last_check >= settings.interval_seconds or status.last_check_time == 0.0:
                    await self._perform_check(provider_name, settings)
                
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
        
        async with self._get_lock(provider_name):
            status = self.get_status(provider_name)
            status.last_check_time = time.time()
            old_state = status.state
            
            if is_healthy:
                status.last_latency = latency
                status.last_success_time = time.time()
                status.consecutive_successes = min(10000, status.consecutive_successes + 1)
                status.consecutive_failures = 0
                status.last_error_category = None
                
                if status.state == ProviderHealthState.UNHEALTHY or status.state == ProviderHealthState.RECOVERING:
                    status.state = ProviderHealthState.RECOVERING
                    if status.consecutive_successes >= settings.healthy_threshold:
                        status.state = ProviderHealthState.HEALTHY
                else:
                    status.state = ProviderHealthState.HEALTHY
                    
            else:
                status.last_failure_time = time.time()
                status.last_error_category = error_category
                status.error_history.append({"time": time.time(), "error": error_category})
                status.consecutive_failures = min(10000, status.consecutive_failures + 1)
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
                    status.state = ProviderHealthState.UNHEALTHY
                    
            self._log_transition(provider_name, status, old_state, "active_health_check")

# Global singleton
health_manager = ProviderHealthManager()
