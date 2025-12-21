"""
Pydantic models for API request/response schemas.
"""

from typing import Optional, List
from pydantic import BaseModel, Field


# Request Models

class AnalysisRequest(BaseModel):
    """Request model for video analysis."""
    video_path: str = Field(..., description="Path to video file on server")
    user_criteria: str = Field(..., description="Content criteria to search for")
    extract_timestamps: bool = Field(True, description="Extract relevant timestamps")
    use_audio: bool = Field(True, description="Analyze audio track")
    debug: bool = Field(False, description="Save temp chunks to ./tmp/ for debugging")


class SimpleAnalysisRequest(BaseModel):
    """Request for simple video analysis without criteria."""
    video_path: str = Field(..., description="Path to video file")
    prompt: str = Field("Describe what is happening in this video.", description="Analysis prompt")
    use_audio: bool = Field(True, description="Analyze audio track")


# Response Models

class ChunkOfInterest(BaseModel):
    """A relevant segment of the video."""
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    reason: str = Field("", description="Why this segment is relevant")


class Verdict(BaseModel):
    """Verdict on whether content matches criteria."""
    matches_criteria: Optional[bool | str] = Field(None, description="YES/NO/PARTIAL")
    confidence: str = Field("MEDIUM", description="HIGH/MEDIUM/LOW")
    justification: str = Field("", description="Explanation of verdict")


class AnalysisResponse(BaseModel):
    """Full analysis response with verdict and timestamps."""
    success: bool
    analysis: str = ""
    verdict: Optional[Verdict] = None
    chunks_of_interest: List[ChunkOfInterest] = []
    segments_to_trim: List[ChunkOfInterest] = []
    video_duration: float = 0.0
    user_criteria: str = ""
    error: Optional[str] = None


class SimpleAnalysisResponse(BaseModel):
    """Simple analysis response."""
    success: bool
    response: str = ""
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
