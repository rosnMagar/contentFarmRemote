"""
QwenOmniClient: Core model client for Qwen2.5-Omni video+audio processing.

This client provides low-level model access for analyzing videos and audio
using Qwen2.5-Omni's native multimodal capabilities.

For high-level analysis workflows (chunked analysis, content criteria,
verdicts, timestamps), use QwenAnalyzer instead.
"""

import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class QwenOmniClient:
    """Low-level client for Qwen2.5-Omni model inference.
    
    This client handles model loading and basic inference operations.
    Use QwenAnalyzer for high-level analysis workflows.
    """
    
    def __init__(self, 
                 model_name: str = "Qwen/Qwen2.5-Omni-3B",
                 device_map: str = "auto",
                 use_flash_attention: bool = False,
                 video_max_pixels: int = 802816,
                 fps: float = 1.0,
                 max_frames: int = 16):
        """Initialize the Qwen-Omni client.
        
        Args:
            model_name: HuggingFace model ID
            device_map: Device mapping strategy ("auto", "cuda", "cpu")
            use_flash_attention: Whether to use flash attention for efficiency
            video_max_pixels: Max total pixels for video processing (reduces OOM)
            fps: Frames per second to sample from video
            max_frames: Maximum number of frames to extract from video
        """
        # Set video processing limits BEFORE importing qwen_omni_utils
        os.environ["VIDEO_MAX_PIXELS"] = str(video_max_pixels)
        os.environ["VIDEO_TOTAL_PIXELS"] = str(video_max_pixels)
        os.environ["FPS"] = str(fps)
        os.environ["FPS_MAX_FRAMES"] = str(max_frames)
        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
        
        try:
            from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
        except ImportError as e:
            raise ImportError(
                "Transformers with Qwen2.5-Omni support not installed. Run:\n"
                "pip install git+https://github.com/huggingface/transformers@v4.51.3-Qwen2.5-Omni-preview"
            ) from e
        
        try:
            from qwen_omni_utils import process_mm_info
        except ImportError as e:
            raise ImportError(
                "qwen-omni-utils not installed. Run:\n"
                "pip install qwen-omni-utils[decord] accelerate soundfile"
            ) from e
        
        self.process_mm_info = process_mm_info
        
        logger.info(f"Loading Qwen-Omni model: {model_name}")
        
        # Configure GPU allocation
        import torch
        num_gpus = torch.cuda.device_count()
        if num_gpus >= 2:
            max_memory = {0: "0GiB", 1: "15GiB"}
            logger.info("Loading model on GPU 1 only")
        else:
            max_memory = None
        
        try:
            if use_flash_attention:
                self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype="auto",
                    device_map=device_map,
                    max_memory=max_memory,
                    attn_implementation="flash_attention_2",
                    enable_audio_output=False,
                )
            else:
                self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype="auto",
                    device_map=device_map,
                    max_memory=max_memory,
                    enable_audio_output=False,
                )
            
            self.processor = Qwen2_5OmniProcessor.from_pretrained(model_name)
            self.model_name = model_name
            logger.info("Successfully loaded Qwen-Omni model")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _extract_assistant_response(self, full_response: str) -> str:
        """Extract only the assistant's response from the full decoded text.
        
        The batch_decode output includes the full conversation (system, user, assistant).
        This method extracts just the assistant's final response.
        """
        markers = ["assistant\n", "assistant:", "<|assistant|>"]
        
        for marker in markers:
            if marker in full_response:
                parts = full_response.rsplit(marker, 1)
                if len(parts) > 1:
                    return parts[1].strip()
        
        # Fallback: look for conversation patterns
        lines = full_response.split('\n')
        in_assistant = False
        assistant_lines = []
        
        for line in lines:
            if line.strip().lower() == 'assistant':
                in_assistant = True
                continue
            if in_assistant:
                if line.strip().lower() in ['system', 'user']:
                    break
                assistant_lines.append(line)
        
        if assistant_lines:
            return '\n'.join(assistant_lines).strip()
        
        return full_response.strip()
    
    def analyze_video(self, 
                      video_path: str,
                      prompt: str = "Describe what is happening in this video.",
                      use_audio: bool = True,
                      return_audio: bool = False,
                      system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Analyze a video file with native audio+video processing.
        
        Args:
            video_path: Path to the video file
            prompt: Question or instruction about the video
            use_audio: Whether to process audio from the video
            return_audio: Whether to generate audio response
            system_prompt: Optional system prompt override
            
        Returns:
            Dictionary with 'response' key containing the analysis text
        """
        if system_prompt is None:
            system_prompt = (
                "You are an expert video analyst capable of perceiving "
                "auditory and visual inputs. Analyze the video content "
                "including both what you see and hear."
            )
        
        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": str(video_path)},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        
        try:
            text = self.processor.apply_chat_template(
                conversation, 
                add_generation_prompt=True, 
                tokenize=False
            )
            
            audios, images, videos = self.process_mm_info(
                conversation, 
                use_audio_in_video=use_audio
            )
            
            # Log audio processing status
            if use_audio:
                if audios:
                    logger.info(f"Audio processing enabled: extracted {len(audios)} audio stream(s)")
                else:
                    logger.warning("Audio processing enabled but no audio streams found")
            else:
                logger.info("Audio processing disabled")
            
            inputs = self.processor(
                text=text,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=use_audio
            )
            
            inputs = inputs.to(self.model.device).to(self.model.dtype)
            
            if return_audio:
                text_ids, audio = self.model.generate(
                    **inputs,
                    use_audio_in_video=use_audio,
                    return_audio=True
                )
                full_response = self.processor.batch_decode(
                    text_ids, 
                    skip_special_tokens=True, 
                    clean_up_tokenization_spaces=False
                )[0]
                return {"response": self._extract_assistant_response(full_response), "audio": audio}
            else:
                text_ids = self.model.generate(
                    **inputs,
                    use_audio_in_video=use_audio,
                    return_audio=False
                )
                full_response = self.processor.batch_decode(
                    text_ids, 
                    skip_special_tokens=True, 
                    clean_up_tokenization_spaces=False
                )[0]
                return {"response": self._extract_assistant_response(full_response)}
                
        except Exception as e:
            logger.error(f"Error analyzing video: {e}")
            return {"response": f"Error analyzing video: {str(e)}"}
    
    def analyze_audio(self,
                      audio_path: str,
                      prompt: str = "What is being said in this audio?") -> Dict[str, Any]:
        """Analyze an audio file.
        
        Args:
            audio_path: Path to the audio file
            prompt: Question about the audio
            
        Returns:
            Dictionary with 'response' key
        """
        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are an expert audio analyst."}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": str(audio_path)},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        
        try:
            text = self.processor.apply_chat_template(
                conversation, 
                add_generation_prompt=True, 
                tokenize=False
            )
            
            audios, images, videos = self.process_mm_info(conversation)
            
            inputs = self.processor(
                text=text,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=True
            )
            inputs = inputs.to(self.model.device).to(self.model.dtype)
            
            text_ids = self.model.generate(**inputs, return_audio=False)
            full_response = self.processor.batch_decode(
                text_ids, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]
            
            return {"response": self._extract_assistant_response(full_response)}
            
        except Exception as e:
            logger.error(f"Error analyzing audio: {e}")
            return {"response": f"Error analyzing audio: {str(e)}"}

    def generate_text(self, 
                      prompt: str, 
                      system_prompt: Optional[str] = None) -> str:
        """Generate a text-only response (no video/audio input).
        
        Args:
            prompt: The prompt to send
            system_prompt: Optional system prompt
            
        Returns:
            Generated text response
        """
        if system_prompt is None:
            system_prompt = "You are an expert video content analyst."
        
        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            },
        ]
        
        try:
            text = self.processor.apply_chat_template(
                conversation, 
                add_generation_prompt=True, 
                tokenize=False
            )
            
            inputs = self.processor(
                text=text,
                return_tensors="pt",
                padding=True
            )
            inputs = inputs.to(self.model.device).to(self.model.dtype)
            
            text_ids = self.model.generate(**inputs, return_audio=False)
            full_response = self.processor.batch_decode(
                text_ids, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]
            
            return self._extract_assistant_response(full_response)
            
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return f"Error generating response: {str(e)}"
