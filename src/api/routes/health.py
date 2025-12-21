"""
Health and status routes.
"""

from fastapi import APIRouter

from api.models import HealthResponse
from api.dependencies import is_ready

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if is_ready() else "loading",
        model_loaded=is_ready()
    )
