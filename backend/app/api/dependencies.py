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

from app.core.retry_engine import RetryEngine, RetryPolicy
from app.core.config import settings

def get_retry_engine() -> RetryEngine:
    # Uses default system configuration (which can be overridden with env vars/settings)
    # The requirement is just that it is injectable and not hard-coded internally
    # In a full setup, the policy would be loaded from global settings
    policy = RetryPolicy(
        max_attempts=getattr(settings, "RETRY_MAX_ATTEMPTS", 3),
        base_delay=getattr(settings, "RETRY_BASE_DELAY", 1.0),
        max_delay=getattr(settings, "RETRY_MAX_DELAY", 30.0),
        jitter_factor=getattr(settings, "RETRY_JITTER_FACTOR", 0.2),
        conservative_unknown_retry=getattr(settings, "RETRY_CONSERVATIVE_UNKNOWN", False)
    )
    return RetryEngine(policy=policy)


from app.core.failover_engine import FailoverEngine, FailoverPolicy


def get_failover_engine() -> FailoverEngine:
    """Injectable factory for the FailoverEngine singleton.

    Reads FAILOVER_MAX_PROVIDER_ATTEMPTS from settings (default 5).
    Auth errors and invalid requests do NOT failover by default.
    Unsupported-model errors DO failover (guarded by capability check).
    """
    from app.core.failover_engine import get_failover_engine as _get
    policy = FailoverPolicy(
        max_provider_attempts=int(getattr(settings, "FAILOVER_MAX_PROVIDER_ATTEMPTS", 5)),
        failover_on_auth_error=bool(getattr(settings, "FAILOVER_ON_AUTH_ERROR", False)),
        failover_on_invalid_request=bool(getattr(settings, "FAILOVER_ON_INVALID_REQUEST", False)),
        failover_on_unsupported_model=bool(getattr(settings, "FAILOVER_ON_UNSUPPORTED_MODEL", True)),
    )
    retry_engine = get_retry_engine()
    return _get(retry_engine=retry_engine, failover_policy=policy)

