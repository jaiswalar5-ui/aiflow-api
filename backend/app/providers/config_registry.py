import os
import importlib
import inspect
from pathlib import Path
from typing import Dict, List, Optional, Callable
from watchfiles import awatch
import yaml
import asyncio

from app.providers.base import ProviderBase
from app.models.provider_config import ProviderConfig, ProviderRegistryConfig
from app.core.logging import get_logger

logger = get_logger(__name__)


class ConfigLoadError(Exception):
    """Raised when provider config fails to load or validate."""
    pass


class ConfigValidationError(Exception):
    """Raised when provider config validation fails."""
    pass


class ConfigurableProviderRegistry:
    """
    Enhanced provider registry that loads configuration from YAML/JSON files.
    Supports hot-reload and dynamic provider registration.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self._providers: Dict[str, ProviderBase] = {}
        self._configs: Dict[str, ProviderConfig] = {}
        self._config_path = config_path or self._get_default_config_path()
        self._config: Optional[ProviderRegistryConfig] = None
        self._reload_task: Optional[asyncio.Task] = None
        self._reload_callbacks: List[Callable] = []
        self._active_requests = 0
        self._reload_lock = asyncio.Lock()
        
    def _get_default_config_path(self) -> str:
        """Get default config path based on environment."""
        # Check for config directory in backend folder
        backend_dir = Path(__file__).parent.parent.parent
        config_path = backend_dir / "config" / "providers.yaml"
        if config_path.exists():
            return str(config_path)
        
        # Fallback to current directory
        return "config/providers.yaml"
    
    def load_config(self, config_path: Optional[str] = None) -> ProviderRegistryConfig:
        """
        Load and validate provider configuration from YAML file.
        
        Args:
            config_path: Path to config file. If None, uses instance config_path.
            
        Returns:
            Validated ProviderRegistryConfig instance.
            
        Raises:
            ConfigLoadError: If file cannot be read or parsed.
            ConfigValidationError: If configuration is invalid.
        """
        path = config_path or self._config_path
        
        if not os.path.exists(path):
            raise ConfigLoadError(f"Config file not found: {path}")
        
        try:
            with open(path, 'r') as f:
                config_data = yaml.safe_load(f)
                
            if not config_data or 'providers' not in config_data:
                raise ConfigValidationError("Config must contain 'providers' list")
                
            config = ProviderRegistryConfig(**config_data)
            logger.info(f"Successfully loaded config from {path} with {len(config.providers)} providers")
            return config
            
        except yaml.YAMLError as e:
            raise ConfigLoadError(f"Failed to parse YAML config: {e}")
        except Exception as e:
            if isinstance(e, (ConfigLoadError, ConfigValidationError)):
                raise
            raise ConfigValidationError(f"Config validation failed: {e}")
    
    def _import_provider_class(self, adapter_type: str) -> type:
        """
        Dynamically import provider class from module path.
        
        Args:
            adapter_type: Module path (e.g., "app.providers.gemini.GeminiProvider")
            
        Returns:
            Provider class.
            
        Raises:
            ConfigLoadError: If import fails.
        """
        try:
            module_path, class_name = adapter_type.rsplit('.', 1)
            module = importlib.import_module(module_path)
            provider_class = getattr(module, class_name)
            if not inspect.isclass(provider_class) or not issubclass(provider_class, ProviderBase):
                raise ConfigLoadError(
                    f"Provider class '{adapter_type}' must inherit from ProviderBase"
                )
            return provider_class
        except (ImportError, AttributeError, ValueError) as e:
            raise ConfigLoadError(f"Failed to import provider class '{adapter_type}': {e}")
    
    def _create_provider_instance(self, config: ProviderConfig) -> ProviderBase:
        """
        Create provider instance from configuration.
        
        Args:
            config: Provider configuration.
            
        Returns:
            Provider instance.
            
        Raises:
            ConfigLoadError: If instance creation fails.
        """
        try:
            provider_class = self._import_provider_class(config.adapter_type)
            
            # Get API key from environment if env_var_prefix is specified
            api_key = None
            if config.env_var_prefix:
                api_key = os.environ.get(f"{config.env_var_prefix}_API_KEY")
                if not api_key:
                    logger.warning(f"{config.env_var_prefix}_API_KEY not set for provider {config.name}")
            
            # Create instance with timeout and optional parameters
            sig = inspect.signature(provider_class.__init__)
            params = sig.parameters
            
            from typing import Any
            kwargs: dict[str, Any] = {'timeout_seconds': config.timeout_seconds}
            if 'api_key' in params and api_key:
                kwargs['api_key'] = api_key
            if 'supported_models' in params:
                kwargs['supported_models'] = config.supported_models
            
            instance = provider_class(**kwargs)
                
            return instance
            
        except Exception as e:
            raise ConfigLoadError(f"Failed to create provider instance for {config.name}: {e}")
    
    def initialize_from_config(self, config_path: Optional[str] = None) -> None:
        """
        Initialize registry by loading config and creating provider instances.
        
        Args:
            config_path: Path to config file. If None, uses instance config_path.
            
        Raises:
            ConfigLoadError: If config loading or provider creation fails.
            ConfigValidationError: If config validation fails.
        """
        config = self.load_config(config_path)
        self._config = config
        
        # Clear existing providers
        self._providers.clear()
        self._configs.clear()
        
        # Create provider instances for enabled providers
        for provider_config in config.get_enabled_providers():
            try:
                provider = self._create_provider_instance(provider_config)
                self._providers[provider_config.name] = provider
                self._configs[provider_config.name] = provider_config
                logger.info(f"Registered provider: {provider_config.name} with {len(provider_config.supported_models)} models")
            except ConfigLoadError as e:
                logger.error(f"Failed to initialize provider {provider_config.name}: {e}")
                # Continue with other providers
                continue
        
        if not self._providers:
            logger.warning("No providers were successfully initialized")
    
    def register_provider(self, name: str, provider: ProviderBase, config: Optional[ProviderConfig] = None) -> None:
        """Register a provider instance with optional config."""
        self._providers[name] = provider
        if config:
            self._configs[name] = config
    
    def get_provider(self, name: str) -> ProviderBase:
        """Retrieve a provider instance by name. Raises KeyError if not found."""
        if name not in self._providers:
            raise KeyError(f"Provider '{name}' not found in registry.")
        return self._providers[name]
    
    def get_provider_config(self, name: str) -> Optional[ProviderConfig]:
        """Get provider configuration by name."""
        return self._configs.get(name)
    
    def list_providers(self) -> List[str]:
        """Return list of all registered provider names."""
        return list(self._providers.keys())
    
    def list_enabled_providers(self) -> List[str]:
        """Return list of enabled provider names (same as list_providers since only enabled are registered)."""
        return self.list_providers()
    
    def get_provider_by_priority(self) -> List[str]:
        """Return provider names sorted by priority (highest first)."""
        providers_with_priority = [
            (name, self._configs[name].priority if name in self._configs else 0)
            for name in self._providers.keys()
        ]
        providers_with_priority.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in providers_with_priority]
    
    def get_providers_by_capability(self, capability: str) -> List[str]:
        """
        Get providers that have a specific capability.
        
        Args:
            capability: Capability name (streaming, function_calling, vision, parallel_requests)
            
        Returns:
            List of provider names with the capability.
        """
        capable_providers = []
        for name, config in self._configs.items():
            if hasattr(config.capabilities, capability) and getattr(config.capabilities, capability):
                capable_providers.append(name)
        return capable_providers
    
    async def start_hot_reload(self, callback: Optional[Callable] = None) -> None:
        """
        Start hot-reload watcher for config file changes.
        
        Args:
            callback: Optional callback function to call on successful reload.
        """
        if callback:
            self._reload_callbacks.append(callback)
            
        if self._reload_task and not self._reload_task.done():
            logger.warning("Hot-reload already running")
            return
            
        self._reload_task = asyncio.create_task(self._watch_config())
        logger.info(f"Started hot-reload watcher for {self._config_path}")
    
    async def stop_hot_reload(self) -> None:
        """Stop hot-reload watcher."""
        if self._reload_task:
            self._reload_task.cancel()
            try:
                await self._reload_task
            except asyncio.CancelledError:
                pass
            self._reload_task = None
            logger.info("Stopped hot-reload watcher")
    
    async def _watch_config(self) -> None:
        """Watch config file for changes and trigger reload."""
        try:
            async for changes in awatch(self._config_path):
                logger.info(f"Config file changed: {changes}")
                await self._safe_reload()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Error watching config file: {e}")
    
    async def _safe_reload(self) -> None:
        """
        Safely reload configuration without dropping active requests.
        Uses validation to ensure new config is valid before applying.
        """
        async with self._reload_lock:
            if self._active_requests > 0:
                logger.warning(f"Delaying reload: {self._active_requests} active requests")
                # Wait for active requests to complete
                await asyncio.sleep(0.5)
                if self._active_requests > 0:
                    logger.warning("Reload skipped: active requests still in progress")
                    return
            
            try:
                # Validate new config without applying
                new_config = self.load_config()
                
                # Test provider instantiation for all enabled providers
                test_providers = {}
                for provider_config in new_config.get_enabled_providers():
                    try:
                        provider = self._create_provider_instance(provider_config)
                        test_providers[provider_config.name] = provider
                    except ConfigLoadError as e:
                        logger.error(f"Validation failed for provider {provider_config.name}: {e}")
                        raise ConfigValidationError(f"Provider {provider_config.name} failed validation: {e}")
                
                # If validation passes, apply the new config
                old_providers = self._providers.copy()
                old_configs = self._configs.copy()
                
                try:
                    self._config = new_config
                    self._providers.clear()
                    self._configs.clear()
                    
                    for name, provider in test_providers.items():
                        self._providers[name] = provider
                        conf = new_config.get_provider_by_name(name)
                        if conf:
                            self._configs[name] = conf
                    
                    logger.info("Successfully reloaded provider configuration")
                    
                    # Call callbacks
                    for callback in self._reload_callbacks:
                        try:
                            if asyncio.iscoroutinefunction(callback):
                                await callback()
                            else:
                                callback()
                        except Exception as e:
                            logger.error(f"Reload callback failed: {e}")
                            
                except Exception as e:
                    # Rollback on error
                    logger.error(f"Error applying new config, rolling back: {e}")
                    self._providers = old_providers
                    self._configs = old_configs
                    raise
                    
            except (ConfigLoadError, ConfigValidationError) as e:
                logger.error(f"Config reload failed: {e}")
                # Keep old config
                raise
    
    async def increment_active_requests(self) -> None:
        """Increment active request counter for hot-reload safety."""
        self._active_requests += 1
    
    async def decrement_active_requests(self) -> None:
        """Decrement active request counter for hot-reload safety."""
        self._active_requests = max(0, self._active_requests - 1)
    
    def get_active_request_count(self) -> int:
        """Get current number of active requests."""
        return self._active_requests


# Global configurable registry instance
configurable_registry = ConfigurableProviderRegistry()
