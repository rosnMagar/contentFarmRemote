"""
Response parsing utilities for extracting structured data from model outputs.
"""

import json
import re
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def parse_verdict(verdict_text: str) -> Dict[str, Any]:
    """Parse verdict response into structured format.
    
    Args:
        verdict_text: Raw text response from model
        
    Returns:
        Dict with matches_criteria, confidence, and justification
    """
    result = {
        "matches_criteria": None,
        "confidence": "MEDIUM",
        "justification": verdict_text
    }
    
    # Parse VERDICT
    verdict_match = re.search(r'VERDICT:\s*(YES|NO|PARTIAL)', verdict_text, re.IGNORECASE)
    if verdict_match:
        verdict_value = verdict_match.group(1).upper()
        result["matches_criteria"] = verdict_value == "YES"
        if verdict_value == "PARTIAL":
            result["matches_criteria"] = "partial"
    
    # Parse CONFIDENCE
    confidence_match = re.search(r'CONFIDENCE:\s*(HIGH|MEDIUM|LOW)', verdict_text, re.IGNORECASE)
    if confidence_match:
        result["confidence"] = confidence_match.group(1).upper()
    
    # Parse JUSTIFICATION
    justification_match = re.search(r'JUSTIFICATION:\s*(.+?)(?:\n|$)', verdict_text, re.IGNORECASE | re.DOTALL)
    if justification_match:
        result["justification"] = justification_match.group(1).strip()
    
    return result


def extract_timestamps(extraction_text: str) -> Dict[str, Any]:
    """Parse timestamp extraction response into structured format.
    
    Args:
        extraction_text: Raw text response (should contain JSON)
        
    Returns:
        Dict with chunks_of_interest, segments_to_trim, and summary
    """
    result = {
        "chunks_of_interest": [],
        "segments_to_trim": [],
        "summary": ""
    }
    
    # Try to extract JSON from markdown code block
    json_match = re.search(r'```json\s*(.*?)\s*```', extraction_text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(1))
            result.update(parsed)
            return result
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON from code block: {e}")
    
    # Try to parse as raw JSON
    try:
        brace_start = extraction_text.find('{')
        brace_end = extraction_text.rfind('}') + 1
        if brace_start >= 0 and brace_end > brace_start:
            parsed = json.loads(extraction_text[brace_start:brace_end])
            result.update(parsed)
            return result
    except json.JSONDecodeError:
        pass
    
    # Fallback: return raw text as summary
    result["summary"] = extraction_text
    return result
