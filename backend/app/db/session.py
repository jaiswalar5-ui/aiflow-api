from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings

# SQLite requires specific connect_args to prevent issues with thread sharing
connect_args = {"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    settings.DATABASE_URL,
    connect_args=connect_args,
    # pool_pre_ping=True (Recommended for production/postgres)
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
