from fastapi import APIRouter
from app.models.base import HealthResponse
from app.core.config import settings

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Check the health status of the application.
    """
    return HealthResponse(
        status="ok",
        version="1.0.0",
        environment=settings.ENVIRONMENT
    )
