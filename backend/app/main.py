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
    
    # Initialize Providers from config
    from app.providers.config_registry import configurable_registry
    
    try:
        config_path = settings.PROVIDER_CONFIG_PATH if hasattr(settings, 'PROVIDER_CONFIG_PATH') else None
        configurable_registry.initialize_from_config(config_path)
        logger.info(f"Initialized {len(configurable_registry.list_providers())} providers from config")
        
        # Start hot-reload watcher if enabled
        if settings.ENABLE_HOT_RELOAD if hasattr(settings, 'ENABLE_HOT_RELOAD') else True:
            await configurable_registry.start_hot_reload()
            logger.info("Hot-reload enabled for provider configuration")
            
    except Exception as e:
        logger.error(f"Failed to initialize provider registry: {e}")
        # Continue with empty registry - will be caught at runtime
        
    yield
    # Shutdown
    logger.info("Shutting down AIFlow backend...")
    
    # Stop hot-reload watcher
    from app.providers.config_registry import configurable_registry
    await configurable_registry.stop_hot_reload()
    
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
    logger.error(
        "Provider exception.",
        extra={"request_id": getattr(exc, "request_id", None)},
        exc_info=True,
    )
    err_type, status_code, retryable = get_error_type_and_code(exc)
    
    error_resp = ErrorResponse(
        error=ErrorDetails(
            type=err_type,
            message=str(exc),
            code=str(status_code),
            retryable=retryable,
            provider=getattr(exc, 'provider_name', None),
            request_id=getattr(exc, "request_id", None),
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
