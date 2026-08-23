from typing import Dict, List
from app.providers.base import ProviderBase

class ProviderRegistry:
    """
    Registry for managing available AI provider instances.
    """
    def __init__(self):
        self._providers: Dict[str, ProviderBase] = {}
        
    def register_provider(self, name: str, provider: ProviderBase) -> None:
        """Register a provider instance under a specific name."""
        self._providers[name] = provider
        
    def get_provider(self, name: str) -> ProviderBase:
        """Retrieve a provider instance by name. Raises KeyError if not found."""
        if name not in self._providers:
            raise KeyError(f"Provider '{name}' not found in registry.")
        return self._providers[name]
        
    def list_providers(self) -> List[str]:
        """Return a list of all registered provider names."""
        return list(self._providers.keys())

# Global registry instance
provider_registry = ProviderRegistry()
