#!/usr/bin/env python3
"""
Test script for the utility modules.

This script tests the path, text, and validation utility modules.
"""

import os
import sys
import json
import tempfile
import logging
from typing import Dict, Any, List, Optional

from src.utils.path_utils import (
    ensure_directory_exists,
    get_current_timestamp,
    normalize_path,
    get_filename,
    construct_output_path,
    is_valid_file,
    is_valid_directory,
    create_temp_file,
    split_path
)

from src.utils.text_utils import (
    estimate_tokens,
    split_text_into_chunks,
    clean_whitespace,
    extract_sections,
    extract_json_from_text,
    clean_doi,
    count_words
)

from src.utils.validation_utils import (
    validate_file_exists,
    validate_directory_exists,
    validate_file_extension,
    validate_api_key,
    validate_model_name,
    validate_url,
    validate_doi,
    validate_integer_range,
    validate_float_range,
    validate_string_length,
    validate_all
)


def test_path_utils():
    """Test path utilities."""
    print("\n=== Testing Path Utilities ===")
    
    # Test ensure_directory_exists
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = os.path.join(temp_dir, "test_dir")
        ensure_directory_exists(test_dir)
        print(f"Directory created: {os.path.exists(test_dir)}")
    
    # Test get_current_timestamp
    timestamp = get_current_timestamp()
    print(f"Current timestamp: {timestamp}")
    
    # Test normalize_path
    normalized = normalize_path("./test")
    print(f"Normalized path: {normalized}")
    
    # Test get_filename
    filename = get_filename("/path/to/file.txt")
    filename_no_ext = get_filename("/path/to/file.txt", with_extension=False)
    print(f"Filename with extension: {filename}")
    print(f"Filename without extension: {filename_no_ext}")
    
    # Test construct_output_path
    output_path = construct_output_path("test", tempfile.gettempdir(), extension=".txt")
    print(f"Constructed output path: {output_path}")
    
    # Test is_valid_file and is_valid_directory
    with tempfile.NamedTemporaryFile() as temp_file:
        print(f"Valid file: {is_valid_file(temp_file.name)}")
    
    print(f"Valid directory: {is_valid_directory(tempfile.gettempdir())}")
    
    # Test create_temp_file
    temp_file_path = create_temp_file("Test content", ".txt")
    print(f"Temp file created: {os.path.exists(temp_file_path)}")
    os.unlink(temp_file_path)
    
    # Test split_path
    directory, filename, extension = split_path("/path/to/file.txt")
    print(f"Split path: directory={directory}, filename={filename}, extension={extension}")


def test_text_utils():
    """Test text utilities."""
    print("\n=== Testing Text Utilities ===")
    
    # Test estimate_tokens
    text = "This is a test text with multiple words."
    tokens = estimate_tokens(text)
    print(f"Estimated tokens: {tokens}")
    
    # Test split_text_into_chunks
    long_text = "This is paragraph 1.\n\nThis is paragraph 2.\n\nThis is paragraph 3."
    chunks = split_text_into_chunks(long_text, max_tokens=10, chars_per_token=1)
    print(f"Split into {len(chunks)} chunks")
    
    # Test clean_whitespace
    dirty_text = "  This   has \t\textra    whitespace  \n  and line breaks.  "
    clean_text = clean_whitespace(dirty_text)
    print(f"Cleaned text: '{clean_text}'")
    
    # Test extract_sections
    section_text = """1. Introduction
    This is the introduction text.
    
    2. Methods
    This is the methods text.
    
    3. Results
    This is the results text.
    """
    sections = extract_sections(section_text, section_pattern=r'^(?:\d+\.)\s+([A-Z][A-Za-z]+)', min_section_length=10)
    print(f"Extracted sections: {list(sections.keys())}")
    
    # Test extract_json_from_text
    json_text = "Some text before {\"key\": \"value\"} and after."
    json_data = extract_json_from_text(json_text)
    print(f"Extracted JSON: {json_data}")
    
    # Test clean_doi
    doi = "https://doi.org/10.1234/abcd.5678"
    clean_doi_str = clean_doi(doi)
    print(f"Cleaned DOI: {clean_doi_str}")
    
    # Test count_words
    words = count_words("This text has five words.")
    print(f"Word count: {words}")


def test_validation_utils():
    """Test validation utilities."""
    print("\n=== Testing Validation Utilities ===")
    
    # Test validate_file_exists
    with tempfile.NamedTemporaryFile() as temp_file:
        valid, error = validate_file_exists(temp_file.name)
        print(f"File exists validation: {valid}")
    
    # Test validate_directory_exists
    valid, error = validate_directory_exists(tempfile.gettempdir())
    print(f"Directory exists validation: {valid}")
    
    # Test validate_file_extension
    valid, error = validate_file_extension("file.txt", [".txt", ".pdf"])
    print(f"File extension validation: {valid}")
    
    # Test validate_api_key
    valid, error = validate_api_key("sk-1234567890abcdef")
    print(f"API key validation: {valid}")
    
    # Test validate_model_name
    allowed_models = {
        "anthropic": ["claude-3-opus-20240229", "claude-3-sonnet-20240229"],
        "openai": ["gpt-4", "gpt-3.5-turbo"]
    }
    valid, error = validate_model_name("claude-3-opus-20240229", "anthropic", allowed_models)
    print(f"Model name validation: {valid}")
    
    # Test validate_url
    valid, error = validate_url("https://example.com")
    print(f"URL validation: {valid}")
    
    # Test validate_doi
    valid, error = validate_doi("10.1234/abcd.5678")
    print(f"DOI validation: {valid}")
    
    # Test validate_integer_range
    valid, error = validate_integer_range(5, 1, 10)
    print(f"Integer range validation: {valid}")
    
    # Test validate_float_range
    valid, error = validate_float_range(5.5, 1.0, 10.0)
    print(f"Float range validation: {valid}")
    
    # Test validate_string_length
    valid, error = validate_string_length("test", 1, 10)
    print(f"String length validation: {valid}")
    
    # Test validate_all
    validations = [
        (validate_integer_range, [5, 1, 10], {}),
        (validate_string_length, ["test", 1, 10], {}),
        (validate_url, ["https://example.com"], {})
    ]
    errors = validate_all(validations)
    print(f"Multiple validations passed: {len(errors) == 0}")


def main():
    """Main function to run all tests."""
    print("====================================")
    print("Utility Modules Integration Test")
    print("====================================\n")
    
    # Run all tests
    test_path_utils()
    test_text_utils()
    test_validation_utils()
    
    print("\n====================================")
    print("All tests completed")
    print("====================================")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())