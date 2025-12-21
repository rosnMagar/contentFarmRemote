"""
Video analysis routes.
"""

import shutil
import tempfile
import logging
from pathlib import Path

from fastapi import APIRouter, File, UploadFile, Form, HTTPException, BackgroundTasks

from api.models import (
    AnalysisRequest, AnalysisResponse,
    SimpleAnalysisResponse,
    ChunkOfInterest, Verdict
)
from api.dependencies import get_client, get_analyzer, is_ready

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analyze", tags=["analysis"])


@router.post("", response_model=AnalysisResponse)
async def analyze_video(request: AnalysisRequest):
    """
    Analyze a video file with content criteria.
    
    The video must exist on the server at the specified path.
    Returns analysis, verdict, and relevant timestamps.
    """
    if not is_ready():
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    video_path = Path(request.video_path)
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Video not found: {request.video_path}")
    
    analyzer = get_analyzer()
    
    try:
        logger.info(f"Analyzing video: {request.video_path}")
        logger.info(f"Criteria: {request.user_criteria}")
        if request.debug:
            logger.info("Debug mode: temp chunks will be saved to ./tmp/")
        
        result = analyzer.analyze_with_criteria(
            video_path=str(video_path),
            user_criteria=request.user_criteria,
            use_audio=request.use_audio,
            extract_timestamps_flag=request.extract_timestamps,
            debug=request.debug
        )
        
        return _build_analysis_response(result, request.user_criteria)
        
    except Exception as e:
        logger.error(f"Error analyzing video: {e}")
        return AnalysisResponse(
            success=False,
            error=str(e),
            user_criteria=request.user_criteria
        )


@router.post("/upload", response_model=AnalysisResponse)
async def analyze_uploaded_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    user_criteria: str = Form(...),
    extract_timestamps: bool = Form(True),
    use_audio: bool = Form(True),
    debug: bool = Form(False)
):
    """
    Upload and analyze a video file.
    
    The video is temporarily saved, analyzed, and then deleted.
    Set debug=True to preserve temp files for inspection.
    """
    if not is_ready():
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    analyzer = get_analyzer()
    
    # Save uploaded file - use ./tmp/ in debug mode
    if debug:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_dir = Path("./tmp/uploads") / f"video_upload_{timestamp}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_dir = str(temp_dir)
    else:
        temp_dir = tempfile.mkdtemp(prefix="video_upload_")
    
    temp_path = Path(temp_dir) / video.filename
    
    try:
        with open(temp_path, "wb") as f:
            content = await video.read()
            f.write(content)
        
        logger.info(f"Uploaded video saved to: {temp_path}")
        if debug:
            logger.info(f"Debug mode: video will be preserved at {temp_path}")
        
        result = analyzer.analyze_with_criteria(
            video_path=str(temp_path),
            user_criteria=user_criteria,
            use_audio=use_audio,
            extract_timestamps_flag=extract_timestamps,
            debug=debug
        )
        
        # Schedule cleanup (skip in debug mode)
        if not debug:
            background_tasks.add_task(shutil.rmtree, temp_dir, ignore_errors=True)
        else:
            logger.info(f"Debug mode: preserving uploaded video at {temp_dir}")
        
        return _build_analysis_response(result, user_criteria)
        
    except Exception as e:
        if not debug:
            shutil.rmtree(temp_dir, ignore_errors=True)
        logger.error(f"Error analyzing uploaded video: {e}")
        return AnalysisResponse(
            success=False,
            error=str(e),
            user_criteria=user_criteria
        )


@router.post("/simple", response_model=SimpleAnalysisResponse)
async def simple_analysis(
    video_path: str = Form(...),
    prompt: str = Form("Describe what is happening in this video."),
    use_audio: bool = Form(True)
):
    """
    Simple video analysis without criteria matching.
    
    Just returns a description of the video content.
    """
    if not is_ready():
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not Path(video_path).exists():
        raise HTTPException(status_code=404, detail=f"Video not found: {video_path}")
    
    client = get_client()
    
    try:
        result = client.analyze_video(
            video_path=video_path,
            prompt=prompt,
            use_audio=use_audio
        )
        return SimpleAnalysisResponse(
            success=True, 
            response=result.get("response", "")
        )
    except Exception as e:
        return SimpleAnalysisResponse(success=False, error=str(e))


def _build_analysis_response(result: dict, user_criteria: str) -> AnalysisResponse:
    """Build AnalysisResponse from analyzer result."""
    verdict_data = result.get("verdict", {})
    verdict = Verdict(
        matches_criteria=verdict_data.get("matches_criteria"),
        confidence=verdict_data.get("confidence", "MEDIUM"),
        justification=verdict_data.get("justification", "")
    )
    
    chunks = [
        ChunkOfInterest(
            start=c.get("start", 0),
            end=c.get("end", 0),
            reason=c.get("reason", "")
        )
        for c in result.get("chunks_of_interest", [])
    ]
    
    trim_segments = [
        ChunkOfInterest(
            start=c.get("start", 0),
            end=c.get("end", 0),
            reason=c.get("reason", "")
        )
        for c in result.get("segments_to_trim", [])
    ]
    
    return AnalysisResponse(
        success=True,
        analysis=result.get("analysis", ""),
        verdict=verdict,
        chunks_of_interest=chunks,
        segments_to_trim=trim_segments,
        video_duration=result.get("video_duration", 0.0),
        user_criteria=user_criteria
    )
