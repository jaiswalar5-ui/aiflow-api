from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logging import setup_logging, get_logger
from app.api.v1.api import api_router

logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    setup_logging(settings.LOG_LEVEL)
    logger.info("Starting up AIFlow backend...")
    
    # Initialize Providers
    from app.providers.registry import provider_registry
    from app.providers.gemini import GeminiProvider
    from app.providers.groq import GroqProvider
    
    if settings.GEMINI_API_KEY:
        logger.info("Registering GeminiProvider")
        provider_registry.register_provider("gemini", GeminiProvider())
    else:
        logger.warning("GEMINI_API_KEY not set. GeminiProvider will not be registered.")
        
    if settings.GROQ_API_KEY:
        logger.info("Registering GroqProvider")
        provider_registry.register_provider("groq", GroqProvider())
    else:
        logger.warning("GROQ_API_KEY not set. GroqProvider will not be registered.")
        
    yield
    # Shutdown
    logger.info("Shutting down AIFlow backend...")
    # Close connections here

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan,
)

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception handlers
from app.providers.errors import (
    ProviderError, ProviderAuthenticationError, ProviderRateLimitError,
    ProviderInvalidRequestError, ProviderUnsupportedModelError,
    ProviderServerError, ProviderTimeoutError, ProviderNetworkError
)
from app.models.error import ErrorResponse, ErrorDetails

def get_error_type_and_code(exc: Exception):
    if isinstance(exc, ProviderAuthenticationError):
        return "authentication_error", 401, False
    elif isinstance(exc, ProviderRateLimitError):
        return "rate_limit_error", 429, True
    elif isinstance(exc, ProviderInvalidRequestError):
        return "invalid_request_error", 400, False
    elif isinstance(exc, ProviderUnsupportedModelError):
        return "unsupported_model_error", 404, False
    elif isinstance(exc, ProviderTimeoutError):
        return "timeout_error", 504, True
    elif isinstance(exc, ProviderNetworkError):
        return "network_error", 503, True
    elif isinstance(exc, ProviderServerError):
        return "server_error", 502, True
    return "api_error", 500, False

@app.exception_handler(ProviderError)
async def provider_exception_handler(request: Request, exc: ProviderError):
    logger.error(f"Provider exception: {exc}", exc_info=True)
    err_type, status_code, retryable = get_error_type_and_code(exc)
    
    error_resp = ErrorResponse(
        error=ErrorDetails(
            type=err_type,
            message=str(exc),
            code=str(status_code),
            retryable=retryable,
            provider=getattr(exc, 'provider', None),
            request_id=None # Could extract from request if we generate one
        )
    )
    return JSONResponse(status_code=status_code, content=error_resp.model_dump())

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {exc}", exc_info=True)
    
    # Don't intercept FastAPI HTTPExceptions if possible, but we might want them canonical too.
    # For now, wrap unknown exceptions in canonical format.
    error_resp = ErrorResponse(
        error=ErrorDetails(
            type="internal_server_error",
            message="Internal server error",
            code="500",
            retryable=False
        )
    )
    return JSONResponse(status_code=500, content=error_resp.model_dump())

# Include API router
app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/")
async def root():
    return {"message": f"Welcome to {settings.PROJECT_NAME}"}
