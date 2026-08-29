from abc import ABC, abstractmethod
from typing import Optional, Dict
from datetime import datetime, timezone
import json

from app.models.quota import ProviderQuotaState, QuotaResetInfo
from app.core.logging import get_logger

logger = get_logger(__name__)


class QuotaStorageBackend(ABC):
    """Abstract base class for quota state storage backends."""
    
    @abstractmethod
    async def get_provider_state(self, provider_name: str) -> Optional[ProviderQuotaState]:
        """Retrieve quota state for a specific provider."""
        pass
    
    @abstractmethod
    async def save_provider_state(self, state: ProviderQuotaState) -> None:
        """Save quota state for a specific provider."""
        pass
    
    @abstractmethod
    async def delete_provider_state(self, provider_name: str) -> bool:
        """Delete quota state for a specific provider."""
        pass
    
    @abstractmethod
    async def get_all_states(self) -> Dict[str, ProviderQuotaState]:
        """Retrieve all provider states."""
        pass
    
    @abstractmethod
    async def clear_all_states(self) -> None:
        """Clear all provider states."""
        pass


class InMemoryQuotaStorage(QuotaStorageBackend):
    """In-memory implementation of quota storage (for single-instance deployments)."""
    
    def __init__(self):
        self._states: Dict[str, ProviderQuotaState] = {}
    
    async def get_provider_state(self, provider_name: str) -> Optional[ProviderQuotaState]:
        """Retrieve quota state from memory."""
        return self._states.get(provider_name)
    
    async def save_provider_state(self, state: ProviderQuotaState) -> None:
        """Save quota state to memory."""
        self._states[state.provider_name] = state
        logger.debug(f"Saved quota state for provider: {state.provider_name}")
    
    async def delete_provider_state(self, provider_name: str) -> bool:
        """Delete quota state from memory."""
        if provider_name in self._states:
            del self._states[provider_name]
            logger.debug(f"Deleted quota state for provider: {provider_name}")
            return True
        return False
    
    async def get_all_states(self) -> Dict[str, ProviderQuotaState]:
        """Retrieve all states from memory."""
        return self._states.copy()
    
    async def clear_all_states(self) -> None:
        """Clear all states from memory."""
        self._states.clear()
        logger.debug("Cleared all quota states")


class RedisQuotaStorage(QuotaStorageBackend):
    """Redis implementation of quota storage (for multi-instance deployments)."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0", key_prefix: str = "quota:"):
        """
        Initialize Redis storage backend.
        
        Args:
            redis_url: Redis connection URL
            key_prefix: Prefix for all keys in Redis
        """
        try:
            import redis.asyncio as redis
            self.redis = redis.from_url(redis_url, decode_responses=True)
            self.key_prefix = key_prefix
            self._connected = False
        except ImportError:
            logger.warning("Redis package not installed. Redis storage will not be available.")
            self.redis = None
            self._connected = False
    
    async def _ensure_connection(self) -> None:
        """Ensure Redis connection is established."""
        if not self._connected and self.redis:
            try:
                await self.redis.ping()
                self._connected = True
                logger.info("Redis quota storage connected")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                self._connected = False
    
    def _make_key(self, provider_name: str) -> str:
        """Generate Redis key for provider state."""
        return f"{self.key_prefix}{provider_name}"
    
    async def get_provider_state(self, provider_name: str) -> Optional[ProviderQuotaState]:
        """Retrieve quota state from Redis."""
        if not self.redis:
            logger.warning("Redis not available, cannot retrieve state")
            return None
        
        await self._ensure_connection()
        if not self._connected:
            return None
        
        try:
            key = self._make_key(provider_name)
            data = await self.redis.get(key)
            if data:
                state_dict = json.loads(data)
                return ProviderQuotaState(**state_dict)
        except Exception as e:
            logger.error(f"Error retrieving state from Redis: {e}")
        
        return None
    
    async def save_provider_state(self, state: ProviderQuotaState) -> None:
        """Save quota state to Redis."""
        if not self.redis:
            logger.warning("Redis not available, cannot save state")
            return
        
        await self._ensure_connection()
        if not self._connected:
            return
        
        try:
            key = self._make_key(state.provider_name)
            data = state.model_dump_json()
            # Set with 24 hour expiration to prevent stale data
            await self.redis.setex(key, 86400, data)
            logger.debug(f"Saved quota state to Redis for provider: {state.provider_name}")
        except Exception as e:
            logger.error(f"Error saving state to Redis: {e}")
    
    async def delete_provider_state(self, provider_name: str) -> bool:
        """Delete quota state from Redis."""
        if not self.redis:
            return False
        
        await self._ensure_connection()
        if not self._connected:
            return False
        
        try:
            key = self._make_key(provider_name)
            result = await self.redis.delete(key)
            if result:
                logger.debug(f"Deleted quota state from Redis for provider: {provider_name}")
            return result > 0
        except Exception as e:
            logger.error(f"Error deleting state from Redis: {e}")
            return False
    
    async def get_all_states(self) -> Dict[str, ProviderQuotaState]:
        """Retrieve all states from Redis."""
        if not self.redis:
            return {}
        
        await self._ensure_connection()
        if not self._connected:
            return {}
        
        try:
            pattern = f"{self.key_prefix}*"
            keys = await self.redis.keys(pattern)
            states = {}
            
            for key in keys:
                provider_name = key.replace(self.key_prefix, "")
                data = await self.redis.get(key)
                if data:
                    state_dict = json.loads(data)
                    states[provider_name] = ProviderQuotaState(**state_dict)
            
            return states
        except Exception as e:
            logger.error(f"Error retrieving all states from Redis: {e}")
            return {}
    
    async def clear_all_states(self) -> None:
        """Clear all states from Redis."""
        if not self.redis:
            return
        
        await self._ensure_connection()
        if not self._connected:
            return
        
        try:
            pattern = f"{self.key_prefix}*"
            keys = await self.redis.keys(pattern)
            if keys:
                await self.redis.delete(*keys)
                logger.debug(f"Cleared {len(keys)} quota states from Redis")
        except Exception as e:
            logger.error(f"Error clearing states from Redis: {e}")
    
    async def close(self) -> None:
        """Close Redis connection."""
        if self.redis:
            await self.redis.close()
            self._connected = False
            logger.info("Redis quota storage connection closed")


class QuotaStorageFactory:
    """Factory for creating quota storage backends."""
    
    @staticmethod
    def create_storage(backend_type: str = "memory", **kwargs) -> QuotaStorageBackend:
        """
        Create a quota storage backend.
        
        Args:
            backend_type: Type of backend ("memory" or "redis")
            **kwargs: Additional arguments for backend initialization
            
        Returns:
            QuotaStorageBackend instance
        """
        if backend_type == "memory":
            return InMemoryQuotaStorage()
        elif backend_type == "redis":
            return RedisQuotaStorage(**kwargs)
        else:
            raise ValueError(f"Unknown backend type: {backend_type}")


# Global storage instance (can be configured via settings)
_quota_storage: Optional[QuotaStorageBackend] = None


def get_quota_storage() -> QuotaStorageBackend:
    """Get the global quota storage instance."""
    global _quota_storage
    if _quota_storage is None:
        # Lazy import to avoid circular dependency
        import os
        backend_type = os.environ.get("QUOTA_STORAGE_BACKEND", "memory")
        _quota_storage = QuotaStorageFactory.create_storage(backend_type)
    return _quota_storage


def set_quota_storage(storage: QuotaStorageBackend) -> None:
    """Set the global quota storage instance (for testing or custom configs)."""
    global _quota_storage
    _quota_storage = storage