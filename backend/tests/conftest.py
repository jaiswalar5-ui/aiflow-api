import pytest
from app.core.security import generate_api_key, get_password_hash
from app.models.domain import ProjectAPIKey
from app.db.base import Base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from app.api.dependencies import get_db
from app.main import app

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
