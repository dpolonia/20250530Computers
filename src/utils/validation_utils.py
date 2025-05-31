"""
Validation utilities for the paper revision tool.

This module provides utilities for validating different types of data,
including files, paths, API keys, and reference formats.
"""

import os
import re
import json
import logging
import datetime
from typing import List, Dict, Any, Optional, Tuple, Union, Callable


logger = logging.getLogger(__name__)


def validate_file_exists(file_path: str) -> Tuple[bool, str]:
    """
    Validate that a file exists.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Tuple of (valid, error_message)
    """
    if not file_path:
        return False, "File path is empty"
    
    if not os.path.exists(file_path):
        return False, f"File does not exist: {file_path}"
    
    if not os.path.isfile(file_path):
        return False, f"Path is not a file: {file_path}"
    
    if not os.access(file_path, os.R_OK):
        return False, f"File is not readable: {file_path}"
    
    return True, ""


def validate_directory_exists(directory_path: str, check_writable: bool = False) -> Tuple[bool, str]:
    """
    Validate that a directory exists.
    
    Args:
        directory_path: Path to the directory
        check_writable: Whether to check if the directory is writable
        
    Returns:
        Tuple of (valid, error_message)
    """
    if not directory_path:
        return False, "Directory path is empty"
    
    if not os.path.exists(directory_path):
        return False, f"Directory does not exist: {directory_path}"
    
    if not os.path.isdir(directory_path):
        return False, f"Path is not a directory: {directory_path}"
    
    if check_writable and not os.access(directory_path, os.W_OK):
        return False, f"Directory is not writable: {directory_path}"
    
    return True, ""


def validate_file_extension(file_path: str, allowed_extensions: List[str]) -> Tuple[bool, str]:
    """
    Validate that a file has an allowed extension.
    
    Args:
        file_path: Path to the file
        allowed_extensions: List of allowed extensions (with or without dot)
        
    Returns:
        Tuple of (valid, error_message)
    """
    # Ensure extensions have dots
    normalized_extensions = [ext if ext.startswith(".") else f".{ext}" for ext in allowed_extensions]
    
    # Get the file extension
    _, extension = os.path.splitext(file_path)
    
    if extension.lower() not in [ext.lower() for ext in normalized_extensions]:
        return False, f"File has invalid extension: {extension}. Allowed: {', '.join(normalized_extensions)}"
    
    return True, ""


def validate_api_key(api_key: str, min_length: int = 8) -> Tuple[bool, str]:
    """
    Validate an API key.
    
    Args:
        api_key: API key to validate
        min_length: Minimum length for the key
        
    Returns:
        Tuple of (valid, error_message)
    """
    if not api_key:
        return False, "API key is empty"
    
    if len(api_key) < min_length:
        return False, f"API key is too short (minimum {min_length} characters)"
    
    return True, ""


def validate_api_key_format(api_key: str, prefix: str) -> Tuple[bool, str]:
    """
    Validate that an API key has the correct format.
    
    Args:
        api_key: API key to validate
        prefix: Expected prefix for the API key
        
    Returns:
        Tuple of (valid, error_message)
    """
    if not api_key:
        return False, "API key is empty"
    
    if not api_key.startswith(prefix):
        return False, f"API key has invalid format. Expected prefix: {prefix}"
    
    return True, ""


def validate_model_name(
    model_name: str,
    provider: str,
    allowed_models: Dict[str, List[str]]
) -> Tuple[bool, str]:
    """
    Validate that a model name is valid for a provider.
    
    Args:
        model_name: Model name to validate
        provider: Provider name
        allowed_models: Dictionary mapping provider names to lists of allowed models
        
    Returns:
        Tuple of (valid, error_message)
    """
    if provider not in allowed_models:
        return False, f"Invalid provider: {provider}. Allowed: {', '.join(allowed_models.keys())}"
    
    if model_name not in allowed_models[provider]:
        return False, f"Invalid model for {provider}: {model_name}. Allowed: {', '.join(allowed_models[provider])}"
    
    return True, ""


def validate_date_format(date_str: str, format_string: str = "%Y-%m-%d") -> Tuple[bool, str]:
    """
    Validate that a date string has the correct format.
    
    Args:
        date_str: Date string to validate
        format_string: Expected format string
        
    Returns:
        Tuple of (valid, error_message)
    """
    try:
        datetime.datetime.strptime(date_str, format_string)
        return True, ""
    except ValueError:
        return False, f"Invalid date format. Expected: {format_string}"


def validate_email(email: str) -> Tuple[bool, str]:
    """
    Validate an email address.
    
    Args:
        email: Email address to validate
        
    Returns:
        Tuple of (valid, error_message)
    """
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(email_pattern, email):
        return False, "Invalid email address format"
    
    return True, ""


def validate_url(url: str) -> Tuple[bool, str]:
    """
    Validate a URL.
    
    Args:
        url: URL to validate
        
    Returns:
        Tuple of (valid, error_message)
    """
    url_pattern = r'^https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
    if not re.match(url_pattern, url):
        return False, "Invalid URL format"
    
    return True, ""


def validate_doi(doi: str) -> Tuple[bool, str]:
    """
    Validate a DOI.
    
    Args:
        doi: DOI to validate
        
    Returns:
        Tuple of (valid, error_message)
    """
    # Clean the DOI
    doi = doi.strip()
    doi = re.sub(r'^https?://doi\.org/', '', doi)
    doi = re.sub(r'^DOI:\s*', '', doi)
    
    # Validate format
    doi_pattern = r'^10\.\d{4,9}/[-._;()/:A-Z0-9]+$'
    if not re.match(doi_pattern, doi, re.IGNORECASE):
        return False, "Invalid DOI format"
    
    return True, ""


def validate_integer_range(value: int, min_value: int, max_value: int) -> Tuple[bool, str]:
    """
    Validate that an integer is within a range.
    
    Args:
        value: Integer to validate
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        
    Returns:
        Tuple of (valid, error_message)
    """
    if value < min_value:
        return False, f"Value {value} is less than the minimum allowed value ({min_value})"
    
    if value > max_value:
        return False, f"Value {value} is greater than the maximum allowed value ({max_value})"
    
    return True, ""


def validate_float_range(value: float, min_value: float, max_value: float) -> Tuple[bool, str]:
    """
    Validate that a float is within a range.
    
    Args:
        value: Float to validate
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        
    Returns:
        Tuple of (valid, error_message)
    """
    if value < min_value:
        return False, f"Value {value} is less than the minimum allowed value ({min_value})"
    
    if value > max_value:
        return False, f"Value {value} is greater than the maximum allowed value ({max_value})"
    
    return True, ""


def validate_string_length(value: str, min_length: int, max_length: int) -> Tuple[bool, str]:
    """
    Validate that a string has a length within a range.
    
    Args:
        value: String to validate
        min_length: Minimum allowed length
        max_length: Maximum allowed length
        
    Returns:
        Tuple of (valid, error_message)
    """
    if len(value) < min_length:
        return False, f"String length ({len(value)}) is less than the minimum allowed length ({min_length})"
    
    if len(value) > max_length:
        return False, f"String length ({len(value)}) is greater than the maximum allowed length ({max_length})"
    
    return True, ""


def validate_json_schema(data: Dict[str, Any], schema: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate that data conforms to a JSON schema.
    
    Args:
        data: Data to validate
        schema: JSON schema
        
    Returns:
        Tuple of (valid, error_messages)
    """
    try:
        import jsonschema
        validator = jsonschema.Draft7Validator(schema)
        errors = list(validator.iter_errors(data))
        if errors:
            return False, [error.message for error in errors]
        return True, []
    except ImportError:
        logger.warning("jsonschema package not found. Schema validation skipped.")
        return True, ["jsonschema package not found. Schema validation skipped."]
    except Exception as e:
        return False, [str(e)]


def validate_reference_format(reference: str, format_type: str = "bibtex") -> Tuple[bool, str]:
    """
    Validate that a reference has the correct format.
    
    Args:
        reference: Reference to validate
        format_type: Expected format type (bibtex, harvard, ieee)
        
    Returns:
        Tuple of (valid, error_message)
    """
    if format_type.lower() == "bibtex":
        # Basic validation for BibTeX format
        if not re.match(r'@\w+\s*\{[^,]+,(?:\s*\w+\s*=\s*\{[^}]*\},?)+\s*\}', reference, re.DOTALL):
            return False, "Invalid BibTeX format"
    
    elif format_type.lower() == "harvard":
        # Basic validation for Harvard format
        if not re.match(r'[A-Za-z]+(?:,\s+[A-Za-z]+)*(?:\s+et\s+al\.)?(?:\s+\(\d{4}\))', reference):
            return False, "Invalid Harvard format"
    
    elif format_type.lower() == "ieee":
        # Basic validation for IEEE format
        if not re.match(r'\[\d+\]\s+[A-Za-z]+(?:,\s+[A-Za-z\.]+)+', reference):
            return False, "Invalid IEEE format"
    
    else:
        return False, f"Unsupported reference format: {format_type}"
    
    return True, ""


def validate_all(
    validations: List[Tuple[Callable[..., Tuple[bool, str]], List[Any], Dict[str, Any]]]
) -> List[str]:
    """
    Run multiple validations and collect error messages.
    
    Args:
        validations: List of tuples containing (validation_function, args, kwargs)
        
    Returns:
        List of error messages (empty if all validations pass)
    """
    errors = []
    
    for validation_func, args, kwargs in validations:
        valid, error = validation_func(*args, **kwargs)
        if not valid:
            errors.append(error)
    
    return errors