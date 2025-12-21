"""
FastAPI server for Qwen-Omni video analysis.

Run with: uvicorn api_server:app --host 0.0.0.0 --port 8000
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.dependencies import load_models
from api.routes import analysis, health

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    load_models()
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Video Analysis API",
    description="Analyze videos using Qwen2.5-Omni with audio-video integration",
    version="1.0.0",
    lifespan=lifespan
)

# Register routes
app.include_router(health.router)
app.include_router(analysis.router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
