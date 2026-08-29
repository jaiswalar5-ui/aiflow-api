import pytest
from app.core.security import generate_api_key, get_password_hash
from app.models.domain import ProjectAPIKey
from app.db.base import Base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from app.api.dependencies import get_db
from app.main import app
from app.core.quota_storage import set_quota_storage, InMemoryQuotaStorage
from app.core.quota_manager import set_quota_manager, QuotaManager

SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"
test_engine = create_engine(
    SQLALCHEMY_DATABASE_URL, 
    connect_args={"check_same_thread": False},
    poolclass=StaticPool
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

@pytest.fixture(scope="module", autouse=True)
def setup_db():
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)

@pytest.fixture(scope="module", autouse=True)
def setup_quota_system():
    """Initialize quota system for tests."""
    # Use in-memory storage for tests
    test_storage = InMemoryQuotaStorage()
    set_quota_storage(test_storage)
    
    test_quota_manager = QuotaManager(storage=test_storage)
    set_quota_manager(test_quota_manager)
    
    yield
    
    # Cleanup
    from app.core.quota_storage import _quota_storage
    from app.core.quota_manager import _quota_manager
    _quota_storage = None
    _quota_manager = None

@pytest.fixture(scope="module", autouse=True)
def setup_providers():
    """Initialize config-driven provider registry for tests."""
    from app.providers.config_registry import configurable_registry
    import tempfile
    import yaml
    import os
    
    # Create a test config with mock provider
    test_config = {
        "providers": [
            {
                "name": "mock",
                "adapter_type": "app.providers.mock.MockProvider",
                "enabled": True,
                "supported_models": ["mock-gpt", "mock-claude"],
                "priority": 100,
                "timeout_seconds": 30.0,
                "retry_policy": {
                    "max_attempts": 1,
                    "backoff_factor": 1.0,
                    "initial_delay": 0.5,
                    "retryable_errors": []
                },
                "health_check_settings": {
                    "enabled": False,
                    "interval_seconds": 30,
                    "timeout_seconds": 2.0,
                    "unhealthy_threshold": 2,
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
        yaml.dump(test_config, f)
        temp_config_path = f.name
    
    try:
        configurable_registry.initialize_from_config(temp_config_path)
        yield
    finally:
        os.unlink(temp_config_path)
        configurable_registry._providers.clear()
        configurable_registry._configs.clear()

@pytest.fixture
def api_key():
    db = TestingSessionLocal()
    raw_key = generate_api_key()
    hashed_key = get_password_hash(raw_key)
    
    db_key = ProjectAPIKey(
        project_name="Test Project",
        api_key_prefix=raw_key[:15],
        hashed_key=hashed_key,
        is_active=True
    )
    db.add(db_key)
    db.commit()
    db.close()
    
    return raw_key
