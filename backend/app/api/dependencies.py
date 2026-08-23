from typing import Generator
from fastapi import Depends
from app.db.session import SessionLocal

# Dependency injection structure for future services
# e.g., db sessions, external API clients

def get_current_user():
    # Placeholder for auth dependency
    pass

def get_db() -> Generator:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
