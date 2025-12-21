"""
QwenAnalyzer: High-level video analysis orchestration using Qwen-Omni.

This module handles the analysis workflow including:
- Scene-based video chunking with context passing
- Content criteria matching with verdicts
- Per-chunk timestamp extraction with cross-chunk merging
"""

import logging
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

import torch

from .clients.qwen_omni import QwenOmniClient
from .prompt import PromptLoader
from .utils.video_utils import get_video_duration, split_video_by_scenes
from .utils.response_parser import parse_verdict, extract_timestamps as extract_timestamps_from_response
from .utils.timestamp_merger import merge_timestamps

logger = logging.getLogger(__name__)


# Default prompts
DEFAULT_PROMPTS = {
    "Qwen System Prompt": (
        "You are an expert video analyst capable of perceiving "
        "auditory and visual inputs. Analyze the video content "
        "including both what you see and hear."
    ),
    "Qwen Chunk Analysis": (
        "Analyze this video segment (Scene {CHUNK_INDEX}, seconds {CHUNK_START} to {CHUNK_END}).\n\n"
        "USER CRITERIA: {USER_CRITERIA}\n\n"
        "PREVIOUS SCENE SUMMARY:\n{PREVIOUS_CONTEXT}\n\n"
        "1. Describe what you see AND hear\n"
        "2. Identify content matching the criteria\n"
        "3. Provide timestamps (0 to {CHUNK_DURATION}s) for matching content\n\n"
        "End with JSON: {\"timestamps\": [{\"start\": 0, \"end\": 5, \"reason\": \"...\", \"continues_from_previous\": false}]}"
    ),
    "Qwen Verdict": (
        "Based on your analysis, provide a verdict.\n\n"
        "USER'S CONTENT CRITERIA:\n{USER_CRITERIA}\n\n"
        "YOUR ANALYSIS:\n{ANALYSIS}\n\n"
        "Respond: VERDICT: [YES/NO/PARTIAL] CONFIDENCE: [HIGH/MEDIUM/LOW] JUSTIFICATION: [reason]"
    )
}


class QwenAnalyzer:
    """High-level analyzer for video content using Qwen-Omni.
    
    Features:
    - Scene-based chunking with context passing between scenes
    - Per-scene timestamp extraction with automatic merging
    - Content criteria matching with verdicts
    """
    
    def __init__(self, 
                 client: QwenOmniClient,
                 prompt_loader: Optional[PromptLoader] = None,
                 scene_config: Optional[Dict[str, Any]] = None):
        """Initialize the QwenAnalyzer.
        
        Args:
            client: QwenOmniClient instance for model inference
            prompt_loader: Optional PromptLoader for configurable prompts
            scene_config: Scene detection settings from config file
        """
        self.client = client
        self.prompt_loader = prompt_loader
        
        # Scene detection settings (from config or defaults)
        self.scene_config = scene_config or {}
        self.default_threshold = self.scene_config.get("threshold", 30.0)
        self.default_min_duration = self.scene_config.get("min_scene_duration", 5.0)
        self.default_max_duration = self.scene_config.get("max_scene_duration", 30.0)
    
    def _load_prompt(self, prompt_name: str) -> str:
        """Load a prompt template by name from config files."""
        if self.prompt_loader is not None:
            try:
                prompt = self.prompt_loader.get_by_name(prompt_name)
                logger.info(f"Loaded prompt '{prompt_name}' from file ({len(prompt)} chars)")
                return prompt
            except (ValueError, FileNotFoundError) as e:
                logger.warning(f"Could not load prompt '{prompt_name}' from files: {e}")
        
        logger.warning(f"Using DEFAULT built-in prompt for '{prompt_name}'")
        return DEFAULT_PROMPTS.get(prompt_name, "")
    
    def _get_chunk_output_dir(self, debug: bool) -> Optional[str]:
        """Get output directory for video chunks.
        
        When debug=True, returns ./tmp/chunks/video_creation_{timestamp}/
        Otherwise returns None (will use system temp).
        """
        if not debug:
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("./tmp/chunks") / f"video_creation_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Debug mode: saving chunks to {output_dir}")
        return str(output_dir)
    
    def analyze_with_criteria(self,
                              video_path: str,
                              user_criteria: str,
                              use_audio: bool = True,
                              extract_timestamps_flag: bool = True,
                              max_scene_duration: Optional[float] = None,
                              scene_threshold: Optional[float] = None,
                              debug: bool = False) -> Dict[str, Any]:
        """Analyze a video with user-defined content criteria.
        
        Uses scene detection to split video at natural boundaries.
        Settings come from config file, can be overridden per-request.
        
        Args:
            video_path: Path to the video file
            user_criteria: What content the user is looking for
            use_audio: Whether to process audio from the video
            extract_timestamps_flag: Whether to identify relevant time ranges
            max_scene_duration: Max scene duration (uses config default if None)
            scene_threshold: Pixel diff threshold (uses config default if None)
            debug: Save temp chunks to ./tmp/ for debugging
            
        Returns:
            Dictionary with analysis, verdict, and chunks_of_interest
        """
        # Use config defaults if not provided
        max_scene_duration = max_scene_duration or self.default_max_duration
        scene_threshold = scene_threshold or self.default_threshold
        
        logger.info(f"Starting criteria-based analysis of: {video_path}")
        logger.info(f"User criteria: {user_criteria}")
        
        # Get video duration
        video_duration = get_video_duration(video_path)
        logger.info(f"Video duration: {video_duration:.1f} seconds")
        
        # Analyze with scene-based chunking
        if video_duration > max_scene_duration:
            logger.info(f"Using scene-based chunking (max {max_scene_duration}s per scene)")
            analysis_result = self._analyze_with_scenes(
                video_path=video_path,
                user_criteria=user_criteria,
                use_audio=use_audio,
                extract_timestamps=extract_timestamps_flag,
                max_scene_duration=max_scene_duration,
                scene_threshold=scene_threshold,
                debug=debug
            )
        else:
            # Single video (short) - analyze directly
            analysis_result = self._analyze_single_video(
                video_path=video_path,
                user_criteria=user_criteria,
                video_duration=video_duration,
                use_audio=use_audio,
                extract_timestamps=extract_timestamps_flag
            )
        
        # Generate verdict
        logger.info("Generating verdict...")
        verdict = self._generate_verdict(
            user_criteria=user_criteria,
            analysis_text=analysis_result.get("analysis", "")
        )
        
        logger.info(f"Verdict: {verdict['matches_criteria']} (Confidence: {verdict['confidence']})")
        
        # Build final result
        result = {
            "analysis": analysis_result.get("analysis", ""),
            "verdict": verdict,
            "chunk_analyses": analysis_result.get("chunk_analyses", []),
            "video_duration": video_duration,
            "user_criteria": user_criteria,
            "chunks_of_interest": analysis_result.get("chunks_of_interest", []),
            "segments_to_trim": analysis_result.get("segments_to_trim", [])
        }
        
        if result["chunks_of_interest"]:
            logger.info(f"Found {len(result['chunks_of_interest'])} relevant segment(s)")
        
        return result
    
    def _analyze_with_scenes(self,
                             video_path: str,
                             user_criteria: str,
                             use_audio: bool,
                             extract_timestamps: bool,
                             max_scene_duration: float,
                             scene_threshold: float,
                             debug: bool = False) -> Dict[str, Any]:
        """Analyze video using scene-based chunking.
        
        Splits video at natural scene boundaries detected via pixel difference,
        then analyzes each scene with context from previous scenes.
        """
        # Determine output directory
        output_dir = self._get_chunk_output_dir(debug)
        
        # Split video by scenes
        try:
            scene_chunks = split_video_by_scenes(
                video_path,
                output_dir=output_dir,
                threshold=scene_threshold,
                min_scene_duration=self.default_min_duration,
                max_scene_duration=max_scene_duration
            )
        except Exception as e:
            logger.error(f"Scene detection failed: {e}")
            return {"analysis": f"Error: Scene detection failed: {str(e)}"}
        
        if not scene_chunks:
            return {"analysis": "Error: No valid scenes detected"}
        
        system_prompt = self._load_prompt("Qwen System Prompt")
        chunk_prompt_template = self._load_prompt("Qwen Chunk Analysis")
        
        chunk_results = []
        previous_context = "This is the first scene."
        
        for i, (chunk_path, start_time, end_time) in enumerate(scene_chunks):
            scene_duration = end_time - start_time
            logger.info(f"Analyzing scene {i+1}/{len(scene_chunks)} ({start_time:.1f}s-{end_time:.1f}s, {scene_duration:.1f}s)")
            torch.cuda.empty_cache()
            
            # Build prompt with context
            prompt = chunk_prompt_template.replace("{CHUNK_INDEX}", str(i + 1))
            prompt = prompt.replace("{CHUNK_START}", f"{start_time:.1f}")
            prompt = prompt.replace("{CHUNK_END}", f"{end_time:.1f}")
            prompt = prompt.replace("{CHUNK_DURATION}", f"{scene_duration:.1f}")
            prompt = prompt.replace("{USER_CRITERIA}", user_criteria)
            prompt = prompt.replace("{PREVIOUS_CONTEXT}", previous_context[:500])
            
            # Analyze scene
            result = self.client.analyze_video(
                chunk_path,
                prompt=prompt,
                use_audio=use_audio,
                system_prompt=system_prompt
            )
            
            response_text = result.get("response", "")
            
            # Extract timestamps from this scene's response
            scene_timestamps = []
            if extract_timestamps:
                ts_data = extract_timestamps_from_response(response_text)
                scene_timestamps = ts_data.get("timestamps", [])
                if not scene_timestamps:
                    scene_timestamps = ts_data.get("chunks_of_interest", [])
            
            # Store result with actual scene offset for merging
            chunk_results.append({
                "chunk": i + 1,
                "offset": start_time,
                "start_time": start_time,
                "end_time": end_time,
                "analysis": response_text,
                "timestamps": scene_timestamps
            })
            
            # Update context for next scene
            previous_context = self._summarize_for_context(response_text)
            
            logger.info(f"Scene {i+1}: found {len(scene_timestamps)} timestamp(s)")
        
        # Clean up temp files (skip in debug mode)
        if scene_chunks and not debug:
            chunk_dir = Path(scene_chunks[0][0]).parent
            try:
                shutil.rmtree(chunk_dir)
            except Exception as e:
                logger.warning(f"Failed to clean up scene chunks: {e}")
        
        # Merge timestamps across scenes
        merged_timestamps = merge_timestamps(chunk_results, gap_tolerance=1.0)
        
        # Combine analyses
        combined_analysis = self._combine_analyses(chunk_results)
        
        return {
            "analysis": combined_analysis,
            "chunk_analyses": chunk_results,
            "chunks_of_interest": merged_timestamps,
            "segments_to_trim": []
        }
    
    def _analyze_single_video(self,
                              video_path: str,
                              user_criteria: str,
                              video_duration: float,
                              use_audio: bool,
                              extract_timestamps: bool) -> Dict[str, Any]:
        """Analyze a short video (single scene) with timestamp extraction."""
        system_prompt = self._load_prompt("Qwen System Prompt")
        
        prompt = (
            f"Analyze this video. USER CRITERIA: {user_criteria}\n\n"
            f"1. Describe what you see AND hear\n"
            f"2. Identify content matching the criteria\n"
            f"3. Provide timestamps (0 to {video_duration:.1f}s) for matching content\n\n"
            f"End with JSON: {{\"timestamps\": [{{\"start\": 0, \"end\": 5, \"reason\": \"...\"}}]}}"
        )
        
        result = self.client.analyze_video(
            video_path,
            prompt=prompt,
            use_audio=use_audio,
            system_prompt=system_prompt
        )
        
        response_text = result.get("response", "")
        
        # Extract timestamps
        timestamps = []
        if extract_timestamps:
            ts_data = extract_timestamps_from_response(response_text)
            timestamps = ts_data.get("timestamps", ts_data.get("chunks_of_interest", []))
        
        return {
            "analysis": response_text,
            "chunk_analyses": [{
                "chunk": 1,
                "offset": 0,
                "start_time": 0,
                "end_time": video_duration,
                "analysis": response_text,
                "timestamps": timestamps
            }],
            "chunks_of_interest": timestamps
        }
    
    def _summarize_for_context(self, analysis: str) -> str:
        """Create a brief summary for context passing to next scene."""
        if len(analysis) < 400:
            return analysis
        return analysis[:400] + "..."
    
    def _combine_analyses(self, chunk_results: List[Dict]) -> str:
        """Combine scene analyses into a single narrative."""
        if len(chunk_results) == 1:
            return chunk_results[0].get("analysis", "")
        
        # Remove JSON timestamp data from display
        combined_parts = []
        for r in chunk_results:
            analysis = r.get("analysis", "")
            # Remove JSON blocks for cleaner output
            if "{\"timestamps\":" in analysis:
                analysis = analysis.split("{\"timestamps\":")[0].strip()
            combined_parts.append(
                f"[{r['start_time']:.1f}s - {r['end_time']:.1f}s]: {analysis}"
            )
        
        return "\n\n".join(combined_parts)
    
    def _generate_verdict(self, user_criteria: str, analysis_text: str) -> Dict[str, Any]:
        """Generate verdict on whether content matches criteria."""
        verdict_prompt_template = self._load_prompt("Qwen Verdict")
        verdict_prompt = verdict_prompt_template.replace("{USER_CRITERIA}", user_criteria)
        verdict_prompt = verdict_prompt.replace("{ANALYSIS}", analysis_text[:2000])
        
        verdict_response = self.client.generate_text(verdict_prompt)
        return parse_verdict(verdict_response)
