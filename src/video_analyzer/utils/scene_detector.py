"""
Scene detection using pixel difference algorithm.

Detects scene boundaries by comparing consecutive frames and identifying
significant visual changes that indicate a new scene.
Uses adaptive thresholding based on local frame differences.
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

# Default thresholds - tuned for typical videos
DEFAULT_THRESHOLD = 30.0  # Higher = less sensitive (fewer scene cuts)
DEFAULT_MIN_SCENE_DURATION = 5.0  # Minimum 5 seconds per scene
DEFAULT_MAX_SCENE_DURATION = 30.0  # Maximum 30 seconds before forced split


def calculate_frame_difference(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """Calculate difference between two frames using absolute difference.
    
    Args:
        frame1: First frame (BGR image)
        frame2: Second frame (BGR image)
        
    Returns:
        Mean absolute difference (0-255 scale)
    """
    if frame1 is None or frame2 is None:
        return 0.0
    
    # Convert to grayscale for simpler comparison
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Calculate absolute difference and mean
    diff = cv2.absdiff(gray1, gray2)
    return float(np.mean(diff))


def detect_scenes(
    video_path: str,
    threshold: float = DEFAULT_THRESHOLD,
    min_scene_duration: float = DEFAULT_MIN_SCENE_DURATION,
    max_scene_duration: float = DEFAULT_MAX_SCENE_DURATION,
    sample_interval_seconds: float = 0.5
) -> List[Tuple[float, float]]:
    """Detect scene boundaries in a video using pixel difference.
    
    Uses a two-pass approach:
    1. First pass: Collect all frame differences
    2. Second pass: Identify significant changes (above threshold AND 
       significantly above local average)
    
    This prevents splitting on continuous motion that happens to
    exceed the threshold frequently.
    
    Args:
        video_path: Path to the video file
        threshold: Absolute pixel difference threshold for scene change
        min_scene_duration: Minimum duration of a scene (seconds)
        max_scene_duration: Maximum duration of a scene (seconds)
        sample_interval_seconds: Compare frames every N seconds
        
    Returns:
        List of (start_time, end_time) tuples for each scene
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        return []
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps if fps > 0 else 0
    
    logger.info(f"Detecting scenes: {video_duration:.1f}s video, {fps:.1f} fps, threshold={threshold}")
    
    # Calculate sample rate in frames
    sample_rate = max(1, int(sample_interval_seconds * fps))
    
    # First pass: collect all frame differences
    prev_frame = None
    frame_diffs = []  # List of (frame_count, diff)
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % sample_rate == 0:
            if prev_frame is not None:
                diff = calculate_frame_difference(frame, prev_frame)
                frame_diffs.append((frame_count, diff))
            prev_frame = frame.copy()
        
        frame_count += 1
    
    cap.release()
    
    if not frame_diffs:
        return [(0.0, video_duration)]
    
    # Calculate statistics
    all_diffs = [d for _, d in frame_diffs]
    avg_diff = np.mean(all_diffs)
    std_diff = np.std(all_diffs)
    
    # Adaptive threshold: must be above BOTH:
    # 1. The configured threshold
    # 2. average + 1.5 * std (significant outlier)
    adaptive_threshold = max(threshold, avg_diff + 1.5 * std_diff)
    
    logger.info(f"Frame diff stats: avg={avg_diff:.1f}, std={std_diff:.1f}, adaptive_threshold={adaptive_threshold:.1f}")
    
    # Second pass: identify scene boundaries
    min_frames_per_scene = int(min_scene_duration * fps)
    max_frames_per_scene = int(max_scene_duration * fps)
    
    scene_boundaries = [0.0]
    last_scene_frame = 0
    
    for frame_num, diff in frame_diffs:
        frames_since_last_scene = frame_num - last_scene_frame
        
        # Scene change: significant visual difference (adaptive threshold)
        is_scene_change = (diff > adaptive_threshold and 
                           frames_since_last_scene >= min_frames_per_scene)
        
        # Force split only if both max duration reached AND diff is above base threshold
        is_forced_split = (frames_since_last_scene >= max_frames_per_scene and
                           diff > threshold / 2)
        
        if is_scene_change or is_forced_split:
            timestamp = frame_num / fps
            scene_boundaries.append(timestamp)
            last_scene_frame = frame_num
            
            if is_scene_change:
                logger.debug(f"Scene change at {timestamp:.2f}s (diff={diff:.1f} > {adaptive_threshold:.1f})")
            else:
                logger.debug(f"Forced split at {timestamp:.2f}s (max duration)")
    
    # Add final boundary at video end
    if scene_boundaries[-1] < video_duration - 0.5:
        scene_boundaries.append(video_duration)
    
    # Convert boundaries to (start, end) tuples
    scenes = []
    for i in range(len(scene_boundaries) - 1):
        start = scene_boundaries[i]
        end = scene_boundaries[i + 1]
        if end - start >= min_scene_duration:
            scenes.append((start, end))
    
    # If no scenes or all filtered, return whole video
    if not scenes:
        scenes = [(0.0, video_duration)]
    
    # Log summary
    avg_duration = sum(e - s for s, e in scenes) / len(scenes) if scenes else 0
    logger.info(f"Detected {len(scenes)} scene(s), avg duration: {avg_duration:.1f}s")
    
    return scenes


def get_scene_durations(scenes: List[Tuple[float, float]]) -> List[float]:
    """Get durations of all scenes."""
    return [end - start for start, end in scenes]
