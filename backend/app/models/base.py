from pydantic import BaseModel, Field
from typing import Any, Optional

class HealthResponse(BaseModel):
    status: str = Field(..., description="Overall health status")
    version: str = Field(default="1.0.0", description="API version")
    environment: str = Field(..., description="Current environment")

class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Error detail message")
    status_code: Optional[int] = Field(None, description="HTTP status code")
