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
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )

# Include API router
app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/")
async def root():
    return {"message": f"Welcome to {settings.PROJECT_NAME}"}
