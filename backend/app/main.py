from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import time
import logging

from app.config import get_settings
from app.core.registry import EngineRegistry
from app.core.session_manager import SessionManager
from app.core.plugin_manager import PluginManager
from app.api.v1 import data, statistics, ml, models

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")

    # Initialize core components
    app.state.registry = EngineRegistry()
    app.state.session_manager = SessionManager()

    # Load plugins if enabled
    if settings.ENABLE_PLUGINS:
        app.state.plugin_manager = PluginManager(settings.PLUGIN_PATH)
        await app.state.plugin_manager.load_plugins()

    # Initialize ML engines if enabled
    if settings.ENABLE_ML:
        from app.engines.ml import initialize_ml_engines
        await initialize_ml_engines(app.state.registry)

    yield

    # Shutdown
    logger.info("Shutting down application")
    await app.state.session_manager.cleanup()


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description="Open-source web-based statistical & ML platform scalable from lightweight analysis to advanced models with endless extensibility",
    version=settings.APP_VERSION,
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


# Include API routers
app.include_router(data.router, prefix=f"{settings.API_V1_PREFIX}/data", tags=["Data"])
app.include_router(statistics.router, prefix=f"{settings.API_V1_PREFIX}/statistics", tags=["Statistics"])
app.include_router(ml.router, prefix=f"{settings.API_V1_PREFIX}/ml", tags=["Machine Learning"])
app.include_router(models.router, prefix=f"{settings.API_V1_PREFIX}/models", tags=["Models"])


# Root endpoint
@app.get("/")
async def root():
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "operational",
        "features": {
            "statistics": True,
            "ml": settings.ENABLE_ML,
            "plugins": settings.ENABLE_PLUGINS,
            "gpu": settings.ENABLE_GPU
        }
    }


# Health check
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": time.time()
    }


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc) if settings.DEBUG else "An error occurred"
        }
    )
