"""
JSON utilities for the paper revision tool.
"""

import json
import re
from typing import Dict, Any, Union, Optional, List


def extract_json_from_text(text: str) -> str:
    """
    Extract a JSON object or array from text.
    
    This function attempts to extract a valid JSON object or array from text,
    handling various formats including markdown code blocks.
    
    Args:
        text: Text potentially containing a JSON object or array
        
    Returns:
        Extracted JSON string
        
    Raises:
        ValueError: If no JSON object is found
    """
    if not text:
        raise ValueError("Input text is empty")
    
    # Try to find JSON object (starts with { and ends with })
    if '{' in text and '}' in text:
        json_start = text.find('{')
        json_end = text.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            return text[json_start:json_end]
    
    # Try to find JSON array (starts with [ and ends with ])
    if '[' in text and ']' in text:
        json_start = text.find('[')
        json_end = text.rfind(']') + 1
        if json_start >= 0 and json_end > json_start:
            return text[json_start:json_end]
    
    # If no direct JSON found, try to clean markdown code blocks
    if '```json' in text or '```' in text:
        # Remove markdown code blocks
        lines = text.split('\n')
        cleaned_lines = []
        in_code_block = False
        
        for line in lines:
            if line.strip().startswith('```'):
                in_code_block = not in_code_block
                continue
            if not in_code_block and line.strip().startswith('```json'):
                in_code_block = True
                continue
            if not line.strip().startswith('```'):
                cleaned_lines.append(line)
        
        cleaned_text = '\n'.join(cleaned_lines)
        
        # Try again with cleaned text (recursive call)
        try:
            return extract_json_from_text(cleaned_text)
        except ValueError:
            pass
    
    # If still no JSON found, raise an error
    raise ValueError("No JSON object or array found in text")


def parse_json_safely(text: str) -> Dict[str, Any]:
    """
    Safely parse JSON from text, handling potential errors.
    
    Args:
        text: Text containing JSON
        
    Returns:
        Parsed JSON object as a dictionary
        
    Raises:
        ValueError: If the text cannot be parsed as JSON
    """
    try:
        # First try to extract JSON from the text
        json_string = extract_json_from_text(text)
        
        # Then parse the extracted JSON
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error extracting or parsing JSON: {str(e)}")


def ensure_json_list(data: Union[List[Dict[str, Any]], Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Ensure that data is a list of dictionaries.
    
    Args:
        data: Either a dictionary or a list of dictionaries
        
    Returns:
        A list of dictionaries
    """
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        return [data]
    else:
        raise ValueError("Data is neither a dictionary nor a list of dictionaries")


def remove_markdown_formatting(text: str) -> str:
    """
    Remove markdown formatting from text.
    
    Args:
        text: Text with potential markdown formatting
        
    Returns:
        Clean text without markdown formatting
    """
    # Remove code blocks
    text = re.sub(r'```[a-z]*\n[\s\S]*?\n```', '', text)
    
    # Remove headers
    text = re.sub(r'^#{1,6}\s+(.+)$', r'\1', text, flags=re.MULTILINE)
    
    # Remove bold and italic
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    
    # Remove links
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    
    return text