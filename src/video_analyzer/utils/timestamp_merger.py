"""
Timestamp merging utilities for cross-chunk video analysis.

Handles conversion of chunk-relative timestamps to video-relative,
and merging of adjacent/overlapping segments.
"""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def merge_timestamps(
    chunk_results: List[Dict[str, Any]], 
    gap_tolerance: float = 1.0
) -> List[Dict[str, Any]]:
    """Merge adjacent timestamp segments across chunks.
    
    Takes chunk results with timestamps relative to each chunk,
    converts them to video-relative timestamps, and merges
    adjacent segments that are within gap_tolerance of each other.
    
    Args:
        chunk_results: List of chunk analysis results, each containing:
            - chunk: chunk index
            - offset: chunk start time in original video
            - timestamps: list of {start, end, reason} dicts (chunk-relative)
        gap_tolerance: Max gap (seconds) between segments to merge them
        
    Returns:
        List of merged timestamp segments with video-relative times
    """
    all_segments = []
    
    # Collect all segments with video-relative timestamps
    for chunk in chunk_results:
        offset = chunk.get("offset", 0)
        timestamps = chunk.get("timestamps", [])
        
        for ts in timestamps:
            # Convert chunk-relative to video-relative
            video_start = ts.get("start", 0) + offset
            video_end = ts.get("end", 0) + offset
            
            all_segments.append({
                "start": video_start,
                "end": video_end,
                "reason": ts.get("reason", ""),
                "continues": ts.get("continues_from_previous", False),
                "source_chunk": chunk.get("chunk", 0)
            })
    
    if not all_segments:
        return []
    
    # Sort by start time
    all_segments.sort(key=lambda x: x["start"])
    
    # Merge overlapping/adjacent segments
    merged = []
    for seg in all_segments:
        if merged and _should_merge(merged[-1], seg, gap_tolerance):
            # Extend previous segment
            merged[-1]["end"] = max(merged[-1]["end"], seg["end"])
            # Combine reasons if different
            if seg["reason"] and seg["reason"] not in merged[-1]["reason"]:
                merged[-1]["reason"] += f"; {seg['reason']}"
        else:
            # Start new segment (remove internal tracking fields)
            merged.append({
                "start": seg["start"],
                "end": seg["end"],
                "reason": seg["reason"]
            })
    
    logger.info(f"Merged {len(all_segments)} segments into {len(merged)} final segments")
    return merged


def _should_merge(prev: Dict, current: Dict, gap_tolerance: float) -> bool:
    """Determine if two segments should be merged.
    
    Merges if:
    - Segments overlap (current.start <= prev.end)
    - Gap between them is within tolerance
    - Current segment is marked as continuing from previous
    """
    if current.get("continues", False):
        return True
    
    gap = current["start"] - prev["end"]
    return gap <= gap_tolerance


def convert_to_video_relative(
    chunk_timestamps: List[Dict[str, Any]], 
    chunk_offset: float
) -> List[Dict[str, Any]]:
    """Convert chunk-relative timestamps to video-relative.
    
    Args:
        chunk_timestamps: List of {start, end, reason} with chunk-relative times
        chunk_offset: Start time of this chunk in the original video
        
    Returns:
        Same timestamps with video-relative times
    """
    return [
        {
            "start": ts.get("start", 0) + chunk_offset,
            "end": ts.get("end", 0) + chunk_offset,
            "reason": ts.get("reason", ""),
            "continues_from_previous": ts.get("continues_from_previous", False)
        }
        for ts in chunk_timestamps
    ]
