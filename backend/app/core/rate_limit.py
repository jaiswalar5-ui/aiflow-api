import time
from abc import ABC, abstractmethod
from typing import Dict, List
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse

from app.core.config import settings

class RateLimiterBase(ABC):
    @abstractmethod
    def is_rate_limited(self, key: str) -> bool:
        pass

class InMemoryRateLimiter(RateLimiterBase):
    def __init__(self, limit: int, window_seconds: int = 60):
        self.limit = limit
        self.window_seconds = window_seconds
        self._requests: Dict[str, List[float]] = {}

    def is_rate_limited(self, key: str) -> bool:
        now = time.time()
        
        if key not in self._requests:
            self._requests[key] = []
            
        # Clean up old requests
        self._requests[key] = [
            req_time for req_time in self._requests[key] 
            if now - req_time < self.window_seconds
        ]
        
        if len(self._requests[key]) >= self.limit:
            return True
            
        self._requests[key].append(now)
        return False

# Abstract instance for injection
_limiter = InMemoryRateLimiter(limit=settings.RATE_LIMIT_PER_MINUTE)

def check_rate_limit(request: Request, key: str):
    if _limiter.is_rate_limited(key):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded",
            headers={"Retry-After": "60"}
        )
