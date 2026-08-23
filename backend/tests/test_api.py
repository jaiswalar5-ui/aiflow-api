from fastapi.testclient import TestClient
from app.main import app
from app.core.config import settings
from app.core.security import generate_api_key, get_password_hash
from app.models.domain import ProjectAPIKey
from app.db.session import engine
from app.db.base import Base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
import pytest
from app.api.dependencies import get_db

# Setup in-memory db for tests
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

client = TestClient(app)

@pytest.fixture(scope="module", autouse=True)
def setup_db():
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)

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

def test_missing_api_key():
    response = client.post(
        f"{settings.API_V1_STR}/chat/completions",
        json={"messages": [{"role": "user", "content": "Hello"}]}
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "Missing API Key"

def test_invalid_api_key():
    response = client.post(
        f"{settings.API_V1_STR}/chat/completions",
        headers={"Authorization": "Bearer sk-invalid-key"},
        json={"messages": [{"role": "user", "content": "Hello"}]}
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid API Key"

def test_valid_chat_request(api_key):
    from app.providers.registry import provider_registry
    from app.providers.mock import MockProvider
    
    # Register mock provider and use its model
    provider_registry.register_provider("mock", MockProvider())
    
    response = client.post(
        f"{settings.API_V1_STR}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": "mock-gpt",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.7
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert data["model"] == "mock-gpt"
    assert len(data["choices"]) == 1
    assert data["choices"][0]["message"]["content"] == "This is a mock provider response."

def test_malformed_request(api_key):
    response = client.post(
        f"{settings.API_V1_STR}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            # Missing "messages" which is required
            "model": "gpt-4"
        }
    )
    assert response.status_code == 422 # Validation Error

def test_rate_limit(api_key):
    # The default limit is 60. Let's create a specific mock rate limiter limit or hit it.
    # To hit 60 takes time, so we just temporarily monkeypatch the rate limit in testing if we could, 
    # but here we'll just test that we can make a request without 429 initially.
    # Ideally, we'd mock the settings.RATE_LIMIT_PER_MINUTE or _limiter.limit.
    
    # We will import the actual _limiter and change its limit for the test
    from app.core.rate_limit import _limiter
    original_limit = _limiter.limit
    _limiter.limit = 2
    
    try:
        # Request 1 (should pass)
        res1 = client.post(
            f"{settings.API_V1_STR}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"messages": [{"role": "user", "content": "Hello"}]}
        )
        assert res1.status_code == 200
        
        # Request 2 (should pass)
        res2 = client.post(
            f"{settings.API_V1_STR}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"messages": [{"role": "user", "content": "Hello"}]}
        )
        assert res2.status_code == 200
        
        # Request 3 (should fail with 429)
        res3 = client.post(
            f"{settings.API_V1_STR}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"messages": [{"role": "user", "content": "Hello"}]}
        )
        assert res3.status_code == 429
        assert res3.json()["detail"] == "Rate limit exceeded"
        assert res3.headers["Retry-After"] == "60"
        
    finally:
        _limiter.limit = original_limit
