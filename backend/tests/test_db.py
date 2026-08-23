from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.db.base import Base
from app.models.domain import ProviderMetadata

# Use an in-memory SQLite database for testing
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def test_database_models():
    # Create the tables
    Base.metadata.create_all(bind=engine)
    
    db = TestingSessionLocal()
    try:
        # Create a mock provider
        provider = ProviderMetadata(
            name="openai",
            base_url="https://api.openai.com/v1",
            is_active=True,
            supported_features={"chat": True}
        )
        db.add(provider)
        db.commit()
        db.refresh(provider)
        
        # Verify provider was created
        assert provider.id is not None
        assert provider.name == "openai"
        assert provider.supported_features == {"chat": True}
        
    finally:
        db.close()
        # Drop all tables after test
        Base.metadata.drop_all(bind=engine)
