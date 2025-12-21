"""
Shared dependencies and state for API routes.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Global instances (initialized on startup)
_client = None
_analyzer = None
_prompt_loader = None


def get_client():
    """Get the QwenOmniClient instance."""
    return _client


def get_analyzer():
    """Get the QwenAnalyzer instance."""
    return _analyzer


def get_prompt_loader():
    """Get the PromptLoader instance."""
    return _prompt_loader


def load_models():
    """Load models and initialize analyzer on startup."""
    global _client, _analyzer, _prompt_loader
    
    from video_analyzer.config import Config
    from video_analyzer.clients.qwen_omni import QwenOmniClient
    from video_analyzer.qwen_analyzer import QwenAnalyzer
    from video_analyzer.prompt import PromptLoader
    
    logger.info("Loading configuration...")
    config = Config()
    qwen_config = config.get("clients", {}).get("qwen_omni", {})
    scene_config = config.get("clients", {}).get("scene_detection", {})
    
    # Initialize prompt loader
    _prompt_loader = PromptLoader(
        config.get("prompt_dir"), 
        config.get("prompts", [])
    )
    
    logger.info("Loading Qwen-Omni model...")
    _client = QwenOmniClient(
        model_name=qwen_config.get("model", "Qwen/Qwen2.5-Omni-3B"),
        device_map=qwen_config.get("device_map", "auto"),
        use_flash_attention=qwen_config.get("use_flash_attention", False),
        video_max_pixels=qwen_config.get("video_max_pixels", 802816),
        fps=qwen_config.get("fps", 1.0),
        max_frames=qwen_config.get("max_frames", 16)
    )
    
    # Initialize analyzer with scene detection config
    _analyzer = QwenAnalyzer(
        client=_client,
        prompt_loader=_prompt_loader,
        scene_config=scene_config
    )
    
    logger.info("Models loaded successfully!")


def is_ready() -> bool:
    """Check if models are loaded and ready."""
    return _client is not None and _analyzer is not None
