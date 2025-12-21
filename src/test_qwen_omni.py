#!/usr/bin/env python3
"""
Test script for Qwen2.5-Omni native video+audio analysis with content criteria.

Usage:
    python test_qwen_omni.py <video_path> [options]

Examples:
    # Basic analysis with default prompt
    python test_qwen_omni.py ../videos/test.mp4
    
    # Analysis with content criteria for searching specific content
    python test_qwen_omni.py ../videos/test.mp4 --criteria "people cheating or being unfaithful"
    
    # With timestamp extraction for longer videos
    python test_qwen_omni.py ../videos/long.mp4 --criteria "funny moments" --timestamps
"""

import sys
import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze videos with Qwen2.5-Omni (audio+video integration)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s ../videos/5.mp4 --criteria "dramatic or emotional content"
  %(prog)s ../videos/5.mp4 --criteria "people talking" --timestamps
  %(prog)s ../videos/5.mp4 --prompt "What is happening?" (legacy mode)
        """
    )
    parser.add_argument("video_path", type=str, help="Path to the video file")
    parser.add_argument("--criteria", "-c", type=str, 
                        help="Content criteria to search for")
    parser.add_argument("--prompt", "-p", type=str,
                        help="Custom prompt (legacy mode, ignored if --criteria is set)")
    parser.add_argument("--timestamps", "-t", action="store_true",
                        help="Extract timestamps for relevant content segments")
    parser.add_argument("--no-audio", action="store_true",
                        help="Disable audio processing")
    parser.add_argument("--chunk-duration", type=int, default=None,
                        help="Duration of each chunk in seconds")
    parser.add_argument("--output", "-o", type=str,
                        help="Save results to JSON file")
    
    args = parser.parse_args()
    
    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)
    
    logger.info("Loading config and initializing clients...")
    
    try:
        from video_analyzer.config import Config
        from video_analyzer.clients.qwen_omni import QwenOmniClient
        from video_analyzer.qwen_analyzer import QwenAnalyzer
        from video_analyzer.prompt import PromptLoader
        
        # Load config
        config = Config()
        qwen_config = config.get("clients", {}).get("qwen_omni", {})
        
        # Initialize prompt loader
        prompt_loader = PromptLoader(
            config.get("prompt_dir"), 
            config.get("prompts", [])
        )
        
        # Initialize client
        client = QwenOmniClient(
            model_name=qwen_config.get("model", "Qwen/Qwen2.5-Omni-3B"),
            device_map=qwen_config.get("device_map", "auto"),
            use_flash_attention=qwen_config.get("use_flash_attention", False),
            video_max_pixels=qwen_config.get("video_max_pixels", 802816),
            fps=qwen_config.get("fps", 1.0),
            max_frames=qwen_config.get("max_frames", 16)
        )
        
        # Initialize analyzer
        chunk_duration = args.chunk_duration or qwen_config.get("chunk_duration", 10)
        analyzer = QwenAnalyzer(
            client=client,
            prompt_loader=prompt_loader,
            chunk_duration=chunk_duration
        )
        
        use_audio = not args.no_audio
        
        logger.info(f"Analyzing video: {video_path}")
        
        # Decide which analysis mode to use
        if args.criteria:
            # New criteria-based analysis with verdict and timestamps
            logger.info(f"Content criteria: {args.criteria}")
            logger.info(f"Timestamp extraction: {args.timestamps}")
            
            result = analyzer.analyze_with_criteria(
                video_path=str(video_path),
                user_criteria=args.criteria,
                use_audio=use_audio,
                extract_timestamps_flag=args.timestamps
            )
            
            # Display results
            print("\n" + "="*60)
            print("VIDEO ANALYSIS RESULT")
            print("="*60)
            print(result.get("analysis", "No analysis generated"))
            
            print("\n" + "="*60)
            print("VERDICT")
            print("="*60)
            verdict = result.get("verdict", {})
            matches = verdict.get("matches_criteria")
            if matches is True:
                print("✅ MATCHES CRITERIA: YES")
            elif matches == "partial":
                print("⚠️  MATCHES CRITERIA: PARTIAL")
            else:
                print("❌ MATCHES CRITERIA: NO")
            print(f"Confidence: {verdict.get('confidence', 'N/A')}")
            print(f"Justification: {verdict.get('justification', 'N/A')}")
            
            # Show timestamps if extracted
            if args.timestamps and result.get("chunks_of_interest"):
                print("\n" + "="*60)
                print("RELEVANT SEGMENTS (for trimming)")
                print("="*60)
                for chunk in result["chunks_of_interest"]:
                    print(f"  [{chunk['start']:.1f}s - {chunk['end']:.1f}s]: {chunk.get('reason', '')}")
                
                if result.get("segments_to_trim"):
                    print("\nSEGMENTS TO TRIM (irrelevant content):")
                    for chunk in result["segments_to_trim"]:
                        print(f"  [{chunk['start']:.1f}s - {chunk['end']:.1f}s]: {chunk.get('reason', '')}")
            
            # Show chunk details
            if result.get("chunk_analyses"):
                print("\n" + "-"*60)
                print("CHUNK DETAILS")
                print("-"*60)
                for chunk in result["chunk_analyses"]:
                    print(f"\n[Chunk {chunk['chunk']} ({chunk['start_time']}-{chunk['end_time']}s)]:")
                    analysis = chunk['analysis']
                    print(analysis[:300] + "..." if len(analysis) > 300 else analysis)
            
        else:
            # Legacy mode: simple prompt-based analysis
            prompt = args.prompt or "Describe what is happening in this video, including any speech or sounds you hear."
            logger.info(f"Using legacy mode with prompt: {prompt}")
            
            result = analyzer.analyze_chunked(
                video_path=str(video_path),
                prompt=prompt,
                use_audio=use_audio
            )
            
            print("\n" + "="*60)
            print("VIDEO ANALYSIS RESULT")
            print("="*60)
            print(result.get("response", "No response generated"))
            
            if result.get("chunk_analyses"):
                print("\n" + "-"*60)
                print("CHUNK DETAILS")
                print("-"*60)
                for chunk in result["chunk_analyses"]:
                    print(f"\n[Chunk {chunk['chunk']} ({chunk['start_time']}-{chunk['end_time']}s)]:")
                    analysis = chunk['analysis']
                    print(analysis[:200] + "..." if len(analysis) > 200 else analysis)
        
        print("="*60)
        
        # Save to file if requested
        if args.output:
            output_path = Path(args.output)
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"Results saved to: {output_path}")
        
    except ImportError as e:
        print(f"\nError: Missing dependencies. Please install required packages:\n")
        print("pip install git+https://github.com/huggingface/transformers@v4.51.3-Qwen2.5-Omni-preview")
        print("pip install qwen-omni-utils[decord] accelerate soundfile")
        print(f"\nOriginal error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        raise


if __name__ == "__main__":
    main()
