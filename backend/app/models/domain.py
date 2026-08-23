import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Boolean, JSON, Integer, Float, DateTime, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID

from app.db.base import Base

def get_uuid():
    return str(uuid.uuid4())

def get_utc_now():
    return datetime.now(timezone.utc)

class ProviderMetadata(Base):
    __tablename__ = "provider_metadata"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=get_uuid, index=True)
    name: Mapped[str] = mapped_column(String, unique=True, index=True)
    base_url: Mapped[str] = mapped_column(String, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    supported_features: Mapped[dict] = mapped_column(JSON, nullable=True)
    
    state = relationship("ProviderState", back_populates="provider", uselist=False)


class ProviderState(Base):
    __tablename__ = "provider_state"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=get_uuid, index=True)
    provider_id: Mapped[str] = mapped_column(String, ForeignKey("provider_metadata.id"), unique=True)
    is_healthy: Mapped[bool] = mapped_column(Boolean, default=True)
    current_quota_usage: Mapped[int] = mapped_column(Integer, default=0)
    last_health_check: Mapped[datetime] = mapped_column(DateTime, default=get_utc_now)
    
    provider = relationship("ProviderMetadata", back_populates="state")


class UsageMetrics(Base):
    __tablename__ = "usage_metrics"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=get_uuid, index=True)
    project_id: Mapped[str] = mapped_column(String, nullable=True, index=True)
    provider_id: Mapped[str] = mapped_column(String, ForeignKey("provider_metadata.id"))
    tokens_used: Mapped[int] = mapped_column(Integer, default=0)
    request_latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=get_utc_now, index=True)


class ProjectAPIKey(Base):
    __tablename__ = "project_api_keys"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=get_uuid, index=True)
    project_name: Mapped[str] = mapped_column(String, index=True)
    api_key_prefix: Mapped[str] = mapped_column(String, index=True)
    hashed_key: Mapped[str] = mapped_column(String)  # Never store raw key
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=get_utc_now)
