"""
Text utilities for the paper revision tool.

This module provides utilities for text processing, including chunking, tokenization,
cleaning, and normalization.
"""

import re
import json
import logging
from typing import List, Dict, Any, Optional, Tuple, Union, Callable, Match


logger = logging.getLogger(__name__)


def estimate_tokens(text: str, chars_per_token: int = 4) -> int:
    """
    Estimate the number of tokens in a text.
    
    Args:
        text: Text to estimate
        chars_per_token: Average number of characters per token
        
    Returns:
        Estimated number of tokens
    """
    return len(text) // chars_per_token


def split_text_into_chunks(
    text: str,
    max_tokens: int = 2000,
    chars_per_token: int = 4,
    overlap: int = 100
) -> List[str]:
    """
    Split text into chunks that fit within a token limit.
    
    Args:
        text: Text to split
        max_tokens: Maximum tokens per chunk
        chars_per_token: Average number of characters per token
        overlap: Number of tokens to overlap between chunks
        
    Returns:
        List of text chunks
    """
    # Convert token limits to character limits
    max_chars = max_tokens * chars_per_token
    overlap_chars = overlap * chars_per_token
    
    # First, split by paragraphs
    paragraphs = re.split(r'\n\s*\n', text)
    
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        # If the paragraph itself is too long, split by sentences
        if len(paragraph) > max_chars:
            sentences = re.split(r'(?<=[.!?])\s+', paragraph)
            for sentence in sentences:
                if len(current_chunk) + len(sentence) + 2 <= max_chars:
                    if current_chunk:
                        current_chunk += "\n\n"
                    current_chunk += sentence
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                        # Keep some overlap for context
                        if overlap_chars > 0 and len(current_chunk) > overlap_chars:
                            current_chunk = current_chunk[-overlap_chars:]
                        else:
                            current_chunk = ""
                    
                    # If the sentence itself is too long, just add it as its own chunk
                    if len(sentence) > max_chars:
                        chunks.append(sentence[:max_chars])
                        current_chunk = sentence[-min(overlap_chars, len(sentence)):]
                    else:
                        current_chunk = sentence
        else:
            # Check if adding this paragraph exceeds the limit
            if len(current_chunk) + len(paragraph) + 2 <= max_chars:
                if current_chunk:
                    current_chunk += "\n\n"
                current_chunk += paragraph
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                    # Keep some overlap for context
                    if overlap_chars > 0 and len(current_chunk) > overlap_chars:
                        current_chunk = current_chunk[-overlap_chars:]
                    else:
                        current_chunk = ""
                current_chunk += paragraph
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks


def clean_whitespace(text: str) -> str:
    """
    Clean whitespace in text.
    
    Args:
        text: Text to clean
        
    Returns:
        Cleaned text
    """
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Normalize line breaks
    text = re.sub(r'\r\n', '\n', text)
    
    # Remove spaces at the beginning and end of lines
    text = re.sub(r'^ +| +$', '', text, flags=re.MULTILINE)
    
    # Remove empty lines
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    return text.strip()


def normalize_newlines(text: str) -> str:
    """
    Normalize newlines in text.
    
    Args:
        text: Text to normalize
        
    Returns:
        Normalized text
    """
    # Replace all types of line breaks with LF
    return text.replace('\r\n', '\n').replace('\r', '\n')


def extract_sections(
    text: str,
    section_pattern: str = r'^(?:\d+\.?)?(?:\s+)?([A-Z][A-Za-z\s]+)$',
    min_section_length: int = 100
) -> Dict[str, str]:
    """
    Extract sections from text.
    
    Args:
        text: Text to extract sections from
        section_pattern: Regular expression pattern for section headings
        min_section_length: Minimum length of section content to include
        
    Returns:
        Dictionary mapping section names to content
    """
    # Normalize newlines
    text = normalize_newlines(text)
    
    # Compile the pattern
    pattern = re.compile(section_pattern, re.MULTILINE)
    
    # Find all section headings
    matches = list(pattern.finditer(text))
    
    # Extract sections
    sections = {}
    for i, match in enumerate(matches):
        section_name = match.group(1).strip()
        start_pos = match.end()
        
        # Determine end position (start of next section or end of text)
        if i < len(matches) - 1:
            end_pos = matches[i + 1].start()
        else:
            end_pos = len(text)
        
        # Extract section content
        section_content = text[start_pos:end_pos].strip()
        
        # Only include if content is long enough
        if len(section_content) >= min_section_length:
            sections[section_name] = section_content
    
    return sections


def extract_json_from_text(text: str) -> Dict[str, Any]:
    """
    Extract JSON from text.
    
    Args:
        text: Text containing JSON
        
    Returns:
        Extracted JSON as a dictionary
    """
    # Try to find JSON using regular expressions
    json_pattern = r'```json\s*([\s\S]*?)\s*```'
    match = re.search(json_pattern, text)
    if match:
        json_text = match.group(1)
        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            pass
    
    # Try to find JSON between curly braces
    json_pattern = r'({[\s\S]*?})'
    matches = re.finditer(json_pattern, text)
    for match in matches:
        json_text = match.group(1)
        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            continue
    
    # If all else fails, try to extract the largest valid JSON object
    i, j = 0, len(text)
    while i < j:
        try:
            return json.loads(text[i:j])
        except json.JSONDecodeError as e:
            # Handle JSONDecodeError hints about the error location
            if "Expecting property name" in str(e):
                i += 1
            elif "Expecting value" in str(e):
                j -= 1
            else:
                # Narrow the range from both ends
                i += 1
                j -= 1
    
    # If we couldn't extract valid JSON, return an empty dictionary
    return {}


def clean_doi(doi: str) -> str:
    """
    Clean a DOI string.
    
    Args:
        doi: DOI string to clean
        
    Returns:
        Cleaned DOI string
    """
    # Remove any URL prefix
    doi = re.sub(r'^https?://doi\.org/', '', doi)
    
    # Remove any "DOI:" prefix
    doi = re.sub(r'^DOI:\s*', '', doi)
    
    # Remove any whitespace
    doi = doi.strip()
    
    return doi


def extract_citations(text: str) -> List[str]:
    """
    Extract citations from text.
    
    Args:
        text: Text to extract citations from
        
    Returns:
        List of extracted citations
    """
    # Extract citations in Harvard style (Author, Year)
    harvard_pattern = r'\(([A-Za-z\s]+(?:et al\.)?(?:,\s*\d{4})?(?:;\s*[A-Za-z\s]+(?:et al\.)?(?:,\s*\d{4})?)*)\)'
    harvard_matches = re.findall(harvard_pattern, text)
    
    # Extract citations in IEEE style [1] or [1-3]
    ieee_pattern = r'\[(\d+(?:-\d+)?(?:,\s*\d+(?:-\d+)?)*)\]'
    ieee_matches = re.findall(ieee_pattern, text)
    
    # Combine and clean the results
    citations = []
    for match in harvard_matches:
        # Split multiple citations
        for citation in re.split(r';\s*', match):
            citations.append(citation.strip())
    
    for match in ieee_matches:
        # Split multiple citations
        for citation in re.split(r',\s*', match):
            # Expand ranges
            if '-' in citation:
                start, end = map(int, citation.split('-'))
                for i in range(start, end + 1):
                    citations.append(str(i))
            else:
                citations.append(citation.strip())
    
    return citations


def highlight_changes(
    original_text: str,
    new_text: str,
    prefix: str = "[[",
    suffix: str = "]]"
) -> str:
    """
    Highlight changes between original and new text.
    
    Args:
        original_text: Original text
        new_text: New text
        prefix: Prefix for highlighting
        suffix: Suffix for highlighting
        
    Returns:
        New text with changes highlighted
    """
    # This is a simple implementation that doesn't handle all cases
    # For a real implementation, you might want to use a diff algorithm
    
    # Split into lines
    original_lines = original_text.splitlines()
    new_lines = new_text.splitlines()
    
    # Find added or changed lines
    result_lines = []
    for i, line in enumerate(new_lines):
        if i >= len(original_lines) or line != original_lines[i]:
            result_lines.append(f"{prefix}{line}{suffix}")
        else:
            result_lines.append(line)
    
    return "\n".join(result_lines)


def remove_urls(text: str) -> str:
    """
    Remove URLs from text.
    
    Args:
        text: Text to process
        
    Returns:
        Text with URLs removed
    """
    url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
    return re.sub(url_pattern, '', text)


def remove_html_tags(text: str) -> str:
    """
    Remove HTML tags from text.
    
    Args:
        text: Text to process
        
    Returns:
        Text with HTML tags removed
    """
    return re.sub(r'<[^>]+>', '', text)


def count_words(text: str) -> int:
    """
    Count words in text.
    
    Args:
        text: Text to count words in
        
    Returns:
        Number of words
    """
    return len(re.findall(r'\b\w+\b', text))