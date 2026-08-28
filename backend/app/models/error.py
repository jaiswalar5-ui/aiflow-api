from typing import Optional
from pydantic import BaseModel

class ErrorDetails(BaseModel):
    type: str
    message: str
    code: Optional[str] = None
    retryable: bool = False
    provider: Optional[str] = None
    request_id: Optional[str] = None

class ErrorResponse(BaseModel):
    error: ErrorDetails
