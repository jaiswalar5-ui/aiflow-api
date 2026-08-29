import pytest
from datetime import datetime, timezone, timedelta
from app.models.quota import (
    ProviderQuotaState,
    ProviderAvailability,
    QuotaConfidence,
    QuotaResetInfo
)
from app.core.quota_storage import (
    InMemoryQuotaStorage,
    QuotaStorageFactory,
    get_quota_storage,
    set_quota_storage
)
from app.core.quota_manager import QuotaManager, get_quota_manager, set_quota_manager


class TestProviderQuotaState:
    """Test ProviderQuotaState model."""
    
    def test_default_state(self):
        """Test default quota state initialization."""
        state = ProviderQuotaState(provider_name="test-provider")
        
        assert state.provider_name == "test-provider"
        assert state.availability == ProviderAvailability.AVAILABLE
        assert state.quota_confidence == QuotaConfidence.UNKNOWN
        assert state.request_count == 0
        assert state.token_usage == 0
        assert state.rate_limit_events == 0
        assert state.consecutive_failures == 0
    
    def test_is_available_default(self):
        """Test that default state is available."""
        state = ProviderQuotaState(provider_name="test-provider")
        assert state.is_available() is True
    
    def test_is_available_cooldown_active(self):
        """Test that provider in active cooldown is not available."""
        state = ProviderQuotaState(provider_name="test-provider")
        state.availability = ProviderAvailability.COOLDOWN
        state.cooldown_until = datetime.now(timezone.utc) + timedelta(seconds=60)
        
        assert state.is_available() is False
    
    def test_is_available_cooldown_expired(self):
        """Test that provider with expired cooldown becomes available."""
        state = ProviderQuotaState(provider_name="test-provider")
        state.availability = ProviderAvailability.COOLDOWN
        state.cooldown_until = datetime.now(timezone.utc) - timedelta(seconds=10)
        
        assert state.is_available() is True
        assert state.availability == ProviderAvailability.AVAILABLE
        assert state.cooldown_until is None
    
    def test_is_available_exhausted(self):
        """Test that exhausted provider is not available."""
        state = ProviderQuotaState(provider_name="test-provider")
        state.availability = ProviderAvailability.EXHAUSTED
        
        assert state.is_available() is False
    
    def test_record_request(self):
        """Test recording a successful request."""
        state = ProviderQuotaState(provider_name="test-provider")
        state.record_request(tokens_used=100)
        
        assert state.request_count == 1
        assert state.token_usage == 100
        assert state.consecutive_failures == 0
    
    def test_record_request_resets_failures(self):
        """Test that successful request resets consecutive failures."""
        state = ProviderQuotaState(provider_name="test-provider")
        state.consecutive_failures = 2
        state.availability = ProviderAvailability.COOLDOWN
        
        state.record_request(tokens_used=50)
        
        assert state.consecutive_failures == 0
        assert state.availability == ProviderAvailability.AVAILABLE
        assert state.cooldown_until is None
    
    def test_record_rate_limit_cooldown(self):
        """Test recording rate limit puts provider in cooldown."""
        state = ProviderQuotaState(provider_name="test-provider")
        
        state.record_rate_limit(response_details={"status": 429}, cooldown_seconds=60)
        
        assert state.availability == ProviderAvailability.COOLDOWN
        assert state.consecutive_failures == 1
        assert state.rate_limit_events == 1
        assert state.last_quota_failure is not None
        assert state.cooldown_until is not None
    
    def test_record_rate_limit_exhausted(self):
        """Test multiple consecutive failures mark provider as exhausted."""
        state = ProviderQuotaState(provider_name="test-provider")
        
        # Record 3 consecutive failures
        for _ in range(3):
            state.record_rate_limit(response_details={"status": 429}, cooldown_seconds=60)
        
        assert state.availability == ProviderAvailability.EXHAUSTED
        assert state.consecutive_failures == 3
        assert state.rate_limit_events == 3
    
    def test_update_reset_info(self):
        """Test updating quota reset information."""
        state = ProviderQuotaState(provider_name="test-provider")
        reset_info = QuotaResetInfo(
            reset_time=datetime.now(timezone.utc) + timedelta(hours=1),
            requests_remaining=100,
            tokens_remaining=10000
        )
        
        state.update_reset_info(reset_info)
        
        assert state.reset_info == reset_info
        assert state.quota_confidence == QuotaConfidence.CONFIRMED
    
    def test_get_estimated_quota_health_available(self):
        """Test health score for available provider."""
        state = ProviderQuotaState(provider_name="test-provider")
        
        health = state.get_estimated_quota_health()
        
        assert health == 1.0
    
    def test_get_estimated_quota_health_cooldown(self):
        """Test health score for provider in cooldown."""
        state = ProviderQuotaState(provider_name="test-provider")
        state.availability = ProviderAvailability.COOLDOWN
        state.cooldown_until = datetime.now(timezone.utc) + timedelta(minutes=30)
        
        health = state.get_estimated_quota_health()
        
        assert 0.0 < health < 1.0
    
    def test_get_estimated_quota_health_exhausted(self):
        """Test health score for exhausted provider."""
        state = ProviderQuotaState(provider_name="test-provider")
        state.availability = ProviderAvailability.EXHAUSTED
        
        health = state.get_estimated_quota_health()
        
        assert health == 0.0
    
    def test_should_skip_for_quota(self):
        """Test provider skip decision based on quota."""
        state = ProviderQuotaState(provider_name="test-provider")
        
        assert state.should_skip_for_quota() is False
        
        state.availability = ProviderAvailability.COOLDOWN
        state.cooldown_until = datetime.now(timezone.utc) + timedelta(seconds=60)
        
        assert state.should_skip_for_quota() is True
    
    def test_get_retry_after_seconds(self):
        """Test getting retry-after seconds."""
        state = ProviderQuotaState(provider_name="test-provider")
        state.availability = ProviderAvailability.COOLDOWN
        state.cooldown_until = datetime.now(timezone.utc) + timedelta(seconds=120)
        
        retry_after = state.get_retry_after_seconds()
        
        assert retry_after is not None
        assert retry_after > 0


class TestInMemoryQuotaStorage:
    """Test in-memory quota storage backend."""
    
    @pytest.fixture
    def storage(self):
        return InMemoryQuotaStorage()
    
    @pytest.mark.asyncio
    async def test_save_and_get_state(self, storage):
        """Test saving and retrieving provider state."""
        state = ProviderQuotaState(provider_name="test-provider")
        state.request_count = 10
        
        await storage.save_provider_state(state)
        retrieved = await storage.get_provider_state("test-provider")
        
        assert retrieved is not None
        assert retrieved.provider_name == "test-provider"
        assert retrieved.request_count == 10
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_state(self, storage):
        """Test getting non-existent state returns None."""
        state = await storage.get_provider_state("nonexistent")
        assert state is None
    
    @pytest.mark.asyncio
    async def test_delete_state(self, storage):
        """Test deleting provider state."""
        state = ProviderQuotaState(provider_name="test-provider")
        await storage.save_provider_state(state)
        
        deleted = await storage.delete_provider_state("test-provider")
        assert deleted is True
        
        retrieved = await storage.get_provider_state("test-provider")
        assert retrieved is None
    
    @pytest.mark.asyncio
    async def test_delete_nonexistent_state(self, storage):
        """Test deleting non-existent state returns False."""
        deleted = await storage.delete_provider_state("nonexistent")
        assert deleted is False
    
    @pytest.mark.asyncio
    async def test_get_all_states(self, storage):
        """Test retrieving all states."""
        state1 = ProviderQuotaState(provider_name="provider1")
        state2 = ProviderQuotaState(provider_name="provider2")
        
        await storage.save_provider_state(state1)
        await storage.save_provider_state(state2)
        
        all_states = await storage.get_all_states()
        
        assert len(all_states) == 2
        assert "provider1" in all_states
        assert "provider2" in all_states
    
    @pytest.mark.asyncio
    async def test_clear_all_states(self, storage):
        """Test clearing all states."""
        state = ProviderQuotaState(provider_name="test-provider")
        await storage.save_provider_state(state)
        
        await storage.clear_all_states()
        
        all_states = await storage.get_all_states()
        assert len(all_states) == 0


class TestQuotaStorageFactory:
    """Test quota storage factory."""
    
    def test_create_memory_storage(self):
        """Test creating in-memory storage."""
        storage = QuotaStorageFactory.create_storage("memory")
        assert isinstance(storage, InMemoryQuotaStorage)
    
    def test_create_invalid_storage(self):
        """Test creating invalid storage type raises error."""
        with pytest.raises(ValueError):
            QuotaStorageFactory.create_storage("invalid")


class TestQuotaManager:
    """Test quota manager functionality."""
    
    @pytest.fixture
    def manager(self):
        storage = InMemoryQuotaStorage()
        return QuotaManager(storage=storage)
    
    @pytest.mark.asyncio
    async def test_get_provider_state_creates_default(self, manager):
        """Test that getting state creates default if not exists."""
        state = await manager.get_provider_state("new-provider")
        
        assert state is not None
        assert state.provider_name == "new-provider"
        assert state.availability == ProviderAvailability.AVAILABLE
    
    @pytest.mark.asyncio
    async def test_record_request(self, manager):
        """Test recording a successful request."""
        await manager.record_request("test-provider", tokens_used=150)
        
        state = await manager.get_provider_state("test-provider")
        assert state.request_count == 1
        assert state.token_usage == 150
    
    @pytest.mark.asyncio
    async def test_record_rate_limit(self, manager):
        """Test recording a rate limit event."""
        headers = {"Retry-After": "60", "X-RateLimit-Remaining": "0"}
        await manager.record_rate_limit(
            "test-provider",
            status_code=429,
            response_headers=headers
        )
        
        state = await manager.get_provider_state("test-provider")
        assert state.rate_limit_events == 1
        assert state.availability == ProviderAvailability.COOLDOWN
        assert state.consecutive_failures == 1
    
    @pytest.mark.asyncio
    async def test_extract_retry_after(self, manager):
        """Test extracting retry-after from headers."""
        headers = {"Retry-After": "120"}
        retry_after = manager._extract_retry_after(headers)
        
        assert retry_after == 120
    
    @pytest.mark.asyncio
    async def test_extract_reset_info_from_headers(self, manager):
        """Test extracting reset info from headers."""
        headers = {
            "X-RateLimit-Reset": str(int(datetime.now(timezone.utc).timestamp()) + 3600),
            "X-RateLimit-Remaining": "50"
        }
        
        reset_info = manager._extract_reset_info(headers, None)
        
        assert reset_info is not None
        assert reset_info.requests_remaining == 50
    
    @pytest.mark.asyncio
    async def test_is_provider_available(self, manager):
        """Test checking provider availability."""
        # Initially available
        assert await manager.is_provider_available("test-provider") is True
        
        # After rate limit
        await manager.record_rate_limit("test-provider", status_code=429)
        assert await manager.is_provider_available("test-provider") is False
    
    @pytest.mark.asyncio
    async def test_get_available_providers(self, manager):
        """Test getting list of available providers."""
        # Record some states
        await manager.record_request("provider1", tokens_used=10)
        await manager.record_rate_limit("provider2", status_code=429)
        
        available = await manager.get_available_providers(["provider1", "provider2"])
        
        assert "provider1" in available
        assert "provider2" not in available
    
    @pytest.mark.asyncio
    async def test_get_provider_health_score(self, manager):
        """Test getting provider health score."""
        # Default should be 1.0
        score = await manager.get_provider_health_score("test-provider")
        assert score == 1.0
        
        # After rate limit, score should decrease
        await manager.record_rate_limit("test-provider", status_code=429)
        score = await manager.get_provider_health_score("test-provider")
        assert score < 1.0
    
    @pytest.mark.asyncio
    async def test_reset_provider_state(self, manager):
        """Test resetting provider state."""
        await manager.record_rate_limit("test-provider", status_code=429)
        assert await manager.is_provider_available("test-provider") is False
        
        await manager.reset_provider_state("test-provider")
        assert await manager.is_provider_available("test-provider") is True
    
    @pytest.mark.asyncio
    async def test_cleanup_expired_states(self, manager):
        """Test cleanup of expired cooldown states."""
        # Create a provider with expired cooldown
        state = await manager.get_provider_state("test-provider")
        state.availability = ProviderAvailability.COOLDOWN
        state.cooldown_until = datetime.now(timezone.utc) - timedelta(seconds=10)
        await manager.storage.save_provider_state(state)
        
        updated = await manager.cleanup_expired_states()
        
        assert updated >= 1
        state = await manager.get_provider_state("test-provider")
        assert state.availability == ProviderAvailability.AVAILABLE
    
    @pytest.mark.asyncio
    async def test_get_quota_status_for_routing(self, manager):
        """Test getting quota status for routing decisions."""
        await manager.record_request("test-provider", tokens_used=100)
        
        status = await manager.get_quota_status_for_routing("test-provider")
        
        assert status["provider_name"] == "test-provider"
        assert status["available"] is True
        assert status["request_count"] == 1
        assert status["token_usage"] == 100
        assert "health_score" in status


class TestQuotaExhaustionScenarios:
    """Test quota exhaustion and recovery scenarios."""
    
    @pytest.fixture
    def manager(self):
        storage = InMemoryQuotaStorage()
        return QuotaManager(storage=storage)
    
    @pytest.mark.asyncio
    async def test_quota_exhaustion_multiple_failures(self, manager):
        """Test that multiple consecutive failures lead to exhaustion."""
        # Simulate 3 consecutive rate limits
        for i in range(3):
            await manager.record_rate_limit("test-provider", status_code=429)
        
        state = await manager.get_provider_state("test-provider")
        assert state.availability == ProviderAvailability.EXHAUSTED
        assert state.consecutive_failures == 3
    
    @pytest.mark.asyncio
    async def test_cooldown_recovery_after_success(self, manager):
        """Test that successful request recovers from cooldown."""
        # Put in cooldown
        await manager.record_rate_limit("test-provider", status_code=429)
        assert await manager.is_provider_available("test-provider") is False
        
        # Successful request should recover
        await manager.record_request("test-provider", tokens_used=50)
        assert await manager.is_provider_available("test-provider") is True
    
    @pytest.mark.asyncio
    async def test_unknown_quota_no_provider_info(self, manager):
        """Test handling of unknown quota (no provider info)."""
        state = await manager.get_provider_state("test-provider")
        
        assert state.quota_confidence == QuotaConfidence.UNKNOWN
        assert state.reset_info is None
    
    @pytest.mark.asyncio
    async def test_confirmed_quota_from_provider(self, manager):
        """Test confirmed quota from provider response."""
        headers = {
            "X-RateLimit-Reset": str(int(datetime.now(timezone.utc).timestamp()) + 3600),
            "X-RateLimit-Remaining": "100"
        }
        
        await manager.record_rate_limit("test-provider", status_code=429, response_headers=headers)
        
        state = await manager.get_provider_state("test-provider")
        assert state.quota_confidence == QuotaConfidence.CONFIRMED
        assert state.reset_info is not None
        assert state.reset_info.requests_remaining == 100


class TestQuotaManagerIntegration:
    """Test quota manager integration with storage."""
    
    @pytest.mark.asyncio
    async def test_global_quota_manager(self):
        """Test global quota manager instance."""
        manager = get_quota_manager()
        assert manager is not None
        
        # Test basic functionality
        await manager.record_request("test-provider", tokens_used=10)
        state = await manager.get_provider_state("test-provider")
        assert state.request_count == 1
    
    @pytest.mark.asyncio
    async def test_custom_quota_manager(self):
        """Test setting custom quota manager."""
        custom_storage = InMemoryQuotaStorage()
        custom_manager = QuotaManager(storage=custom_storage)
        
        set_quota_manager(custom_manager)
        
        manager = get_quota_manager()
        assert manager.storage == custom_storage