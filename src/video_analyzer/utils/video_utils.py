"""
Video utility functions for duration, splitting, and manipulation.

Uses moviepy for cleaner video processing API.
Scene-based splitting using pixel difference algorithm.
"""

import tempfile
import logging
from pathlib import Path
from typing import List, Optional, Tuple

# Try both moviepy import styles for compatibility
try:
    from moviepy import VideoFileClip
except ImportError:
    from moviepy.editor import VideoFileClip

from .scene_detector import detect_scenes

logger = logging.getLogger(__name__)


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Duration in seconds, or 0.0 if unable to determine
    """
    # Check if file exists first
    if not Path(video_path).exists():
        logger.error(f"Video file not found: {video_path}")
        return 0.0
    
    try:
        with VideoFileClip(video_path) as clip:
            duration = clip.duration
            logger.info(f"Video duration: {duration}s for {video_path}")
            return duration
    except Exception as e:
        logger.error(f"Could not get video duration for {video_path}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 0.0


def split_video_by_scenes(
    video_path: str,
    output_dir: Optional[str] = None,
    threshold: float = 10.0,
    min_scene_duration: float = 5.0,
    max_scene_duration: float = 15.0
) -> List[Tuple[str, float, float]]:
    """Split a video by detected scene boundaries.
    
    Uses pixel difference algorithm to detect scene changes and splits
    the video at natural scene boundaries.
    
    Args:
        video_path: Path to the source video
        output_dir: Directory for chunk files (default: temp dir)
        threshold: Pixel difference threshold for scene detection
        min_scene_duration: Minimum scene duration in seconds
        max_scene_duration: Maximum scene duration in seconds
        
    Returns:
        List of (chunk_path, start_time, end_time) tuples
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="scene_chunks_")
    else:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Detect scenes
    logger.info("Detecting scene boundaries...")
    scenes = detect_scenes(
        video_path,
        threshold=threshold,
        min_scene_duration=min_scene_duration,
        max_scene_duration=max_scene_duration
    )
    
    if not scenes:
        logger.warning("No scenes detected, treating entire video as one scene")
        duration = get_video_duration(video_path)
        scenes = [(0.0, duration)]
    
    logger.info(f"Splitting video into {len(scenes)} scene(s)")
    
    chunk_results = []
    
    try:
        with VideoFileClip(video_path) as clip:
            for i, (start_time, end_time) in enumerate(scenes):
                scene_duration = end_time - start_time
                chunk_path = Path(output_dir) / f"scene_{i:03d}.mp4"
                
                # Extract scene subclip
                subclip = clip.subclipped(start_time, end_time)
                subclip.write_videofile(
                    str(chunk_path),
                    codec="libx264",
                    audio_codec="aac",
                    logger=None
                )
                
                if chunk_path.exists():
                    chunk_results.append((str(chunk_path), start_time, end_time))
                    logger.info(f"Created scene {i+1}/{len(scenes)}: {start_time:.1f}s-{end_time:.1f}s ({scene_duration:.1f}s)")
                    
    except Exception as e:
        logger.error(f"Error splitting video by scenes: {e}")
    
    logger.info(f"Split video into {len(chunk_results)} scene chunks")
    return chunk_results
