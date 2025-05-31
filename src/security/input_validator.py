"""
Input validation utilities for secure data handling.

This module provides functions and decorators to validate user input,
file paths, and other external data before processing.
"""

import os
import re
import logging
from pathlib import Path
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

# Configure logging
logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Exception raised for input validation errors."""
    pass


def validate_file_path(path: Union[str, Path], 
                     file_type: Optional[str] = None,
                     must_exist: bool = True,
                     readable: bool = True) -> Path:
    """
    Validate a file path to ensure it meets security requirements.
    
    Args:
        path: Path to validate (string or Path object)
        file_type: Optional file extension to validate (e.g., "pdf")
        must_exist: Whether the file must exist
        readable: Whether the file must be readable
        
    Returns:
        Path object of validated path
        
    Raises:
        ValidationError: If validation fails
    """
    if not path:
        raise ValidationError("File path cannot be empty")
        
    try:
        # Convert to Path object if string
        if isinstance(path, str):
            path_obj = Path(path)
        else:
            path_obj = path
            
        # Convert to absolute path
        path_obj = path_obj.absolute()
        
        # Check if file exists
        if must_exist and not path_obj.exists():
            raise ValidationError(f"File not found: {path_obj}")
            
        # Check if it's a file (not a directory)
        if must_exist and not path_obj.is_file():
            raise ValidationError(f"Path is not a file: {path_obj}")
            
        # Check file extension
        if file_type and path_obj.suffix.lower() != f".{file_type.lower()}":
            raise ValidationError(
                f"Invalid file type: {path_obj.suffix}. Expected: .{file_type}"
            )
            
        # Check if file is readable
        if must_exist and readable and not os.access(path_obj, os.R_OK):
            raise ValidationError(f"File is not readable: {path_obj}")
            
        return path_obj
        
    except ValidationError:
        raise
    except Exception as e:
        raise ValidationError(f"Invalid file path: {e}")


def validate_directory_path(path: Union[str, Path],
                          must_exist: bool = True,
                          create_if_missing: bool = False,
                          writable: bool = True) -> Path:
    """
    Validate a directory path to ensure it meets security requirements.
    
    Args:
        path: Directory path to validate
        must_exist: Whether the directory must exist
        create_if_missing: Whether to create the directory if it doesn't exist
        writable: Whether the directory must be writable
        
    Returns:
        Path object of validated directory
        
    Raises:
        ValidationError: If validation fails
    """
    if not path:
        raise ValidationError("Directory path cannot be empty")
        
    try:
        # Convert to Path object if string
        if isinstance(path, str):
            path_obj = Path(path)
        else:
            path_obj = path
            
        # Convert to absolute path
        path_obj = path_obj.absolute()
        
        # Create directory if it doesn't exist and creation is allowed
        if not path_obj.exists():
            if create_if_missing:
                try:
                    path_obj.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    raise ValidationError(f"Failed to create directory: {e}")
            elif must_exist:
                raise ValidationError(f"Directory not found: {path_obj}")
                
        # Check if it's a directory (not a file)
        if path_obj.exists() and not path_obj.is_dir():
            raise ValidationError(f"Path is not a directory: {path_obj}")
            
        # Check if directory is writable
        if path_obj.exists() and writable and not os.access(path_obj, os.W_OK):
            raise ValidationError(f"Directory is not writable: {path_obj}")
            
        return path_obj
        
    except ValidationError:
        raise
    except Exception as e:
        raise ValidationError(f"Invalid directory path: {e}")


def validate_string_input(value: Any,
                        min_length: Optional[int] = None,
                        max_length: Optional[int] = None,
                        allowed_chars: Optional[str] = None,
                        pattern: Optional[str] = None,
                        strip: bool = True) -> str:
    """
    Validate a string input to ensure it meets requirements.
    
    Args:
        value: Input value to validate
        min_length: Minimum allowed length
        max_length: Maximum allowed length
        allowed_chars: String of allowed characters
        pattern: Regex pattern the string must match
        strip: Whether to strip whitespace before validation
        
    Returns:
        Validated string
        
    Raises:
        ValidationError: If validation fails
    """
    if value is None:
        raise ValidationError("Input cannot be None")
        
    # Convert to string if needed
    if not isinstance(value, str):
        try:
            value = str(value)
        except Exception as e:
            raise ValidationError(f"Failed to convert input to string: {e}")
            
    # Strip whitespace if requested
    if strip and value:
        value = value.strip()
        
    # Check minimum length
    if min_length is not None and len(value) < min_length:
        raise ValidationError(
            f"Input too short: {len(value)} chars. Minimum: {min_length} chars."
        )
        
    # Check maximum length
    if max_length is not None and len(value) > max_length:
        raise ValidationError(
            f"Input too long: {len(value)} chars. Maximum: {max_length} chars."
        )
        
    # Check allowed characters
    if allowed_chars is not None:
        allowed_set = set(allowed_chars)
        for char in value:
            if char not in allowed_set:
                raise ValidationError(
                    f"Invalid character in input: '{char}'. "
                    f"Allowed characters: {allowed_chars}"
                )
                
    # Check regex pattern
    if pattern is not None:
        if not re.match(pattern, value):
            raise ValidationError(
                f"Input does not match required pattern: {pattern}"
            )
            
    return value


def validate_numeric_input(value: Any,
                         min_value: Optional[float] = None,
                         max_value: Optional[float] = None,
                         allow_float: bool = True) -> Union[int, float]:
    """
    Validate a numeric input to ensure it meets requirements.
    
    Args:
        value: Input value to validate
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        allow_float: Whether to allow floating point values
        
    Returns:
        Validated numeric value (int or float)
        
    Raises:
        ValidationError: If validation fails
    """
    if value is None:
        raise ValidationError("Numeric input cannot be None")
        
    # Convert to numeric type
    if isinstance(value, str):
        value = value.strip()
        try:
            if allow_float:
                value = float(value)
                # Convert to int if the float is a whole number
                if value.is_integer():
                    value = int(value)
            else:
                value = int(value)
        except ValueError:
            raise ValidationError(
                f"Invalid numeric input: '{value}'. "
                f"Expected {'float or integer' if allow_float else 'integer'}"
            )
    elif not isinstance(value, (int, float)):
        raise ValidationError(
            f"Invalid numeric type: {type(value).__name__}. "
            f"Expected {'float or integer' if allow_float else 'integer'}"
        )
        
    # Check if float is allowed
    if not allow_float and isinstance(value, float):
        raise ValidationError(f"Floating point value not allowed: {value}")
        
    # Check minimum value
    if min_value is not None and value < min_value:
        raise ValidationError(
            f"Value too small: {value}. Minimum: {min_value}"
        )
        
    # Check maximum value
    if max_value is not None and value > max_value:
        raise ValidationError(
            f"Value too large: {value}. Maximum: {max_value}"
        )
        
    return value


def validate_choice(value: Any,
                  choices: Union[List, Set, Tuple],
                  case_sensitive: bool = True) -> Any:
    """
    Validate that a value is one of the allowed choices.
    
    Args:
        value: Input value to validate
        choices: Collection of allowed choices
        case_sensitive: Whether to use case-sensitive comparison for strings
        
    Returns:
        Validated choice value
        
    Raises:
        ValidationError: If validation fails
    """
    if not choices:
        raise ValidationError("No choices provided for validation")
        
    # Handle case sensitivity for strings
    if isinstance(value, str) and not case_sensitive:
        value_to_check = value.lower()
        choices_to_check = [
            c.lower() if isinstance(c, str) else c for c in choices
        ]
    else:
        value_to_check = value
        choices_to_check = choices
        
    if value_to_check not in choices_to_check:
        raise ValidationError(
            f"Invalid choice: '{value}'. "
            f"Allowed choices: {', '.join(str(c) for c in choices)}"
        )
        
    # Return the original value, not the lowercase version
    return value


def validate_api_key(api_key: Optional[str], 
                   min_length: int = 8,
                   required: bool = True) -> Optional[str]:
    """
    Validate an API key for basic security requirements.
    
    Args:
        api_key: API key to validate
        min_length: Minimum key length
        required: Whether the key is required
        
    Returns:
        Validated API key
        
    Raises:
        ValidationError: If validation fails
    """
    if not api_key:
        if required:
            raise ValidationError("API key is required")
        return None
        
    # Strip any whitespace
    api_key = api_key.strip()
    
    # Check minimum length
    if len(api_key) < min_length:
        raise ValidationError(
            f"API key too short: {len(api_key)} chars. "
            f"Minimum: {min_length} chars."
        )
        
    # Check for common placeholders
    placeholder_patterns = [
        r'your.?api.?key',
        r'api.?key.?here',
        r'test.?key',
        r'sample.?key',
        r'demo.?key',
        r'<.*>',  # Anything in angle brackets
        r'\[.*\]',  # Anything in square brackets
    ]
    
    for pattern in placeholder_patterns:
        if re.search(pattern, api_key, re.IGNORECASE):
            raise ValidationError(
                f"API key appears to be a placeholder: {api_key}"
            )
            
    return api_key


def validate_doi(doi: str) -> str:
    """
    Validate a DOI (Digital Object Identifier).
    
    Args:
        doi: DOI to validate
        
    Returns:
        Normalized DOI string
        
    Raises:
        ValidationError: If validation fails
    """
    if not doi:
        raise ValidationError("DOI cannot be empty")
        
    # Strip whitespace
    doi = doi.strip()
    
    # Remove 'doi:' or 'DOI:' prefix if present
    if doi.lower().startswith('doi:'):
        doi = doi[4:].strip()
        
    # Remove 'https://doi.org/' prefix if present
    if doi.lower().startswith('https://doi.org/'):
        doi = doi[16:].strip()
        
    # Basic DOI format validation
    # DOIs typically follow the pattern 10.XXXX/YYYY
    doi_pattern = r'^10\.\d{4,9}/[-._;()/:A-Z0-9]+$'
    
    if not re.match(doi_pattern, doi, re.IGNORECASE):
        raise ValidationError(
            f"Invalid DOI format: {doi}. "
            "Expected format: 10.XXXX/YYYY"
        )
        
    return doi


def validate_email(email: str) -> str:
    """
    Validate an email address.
    
    Args:
        email: Email address to validate
        
    Returns:
        Validated email address
        
    Raises:
        ValidationError: If validation fails
    """
    if not email:
        raise ValidationError("Email cannot be empty")
        
    # Strip whitespace
    email = email.strip()
    
    # Basic email format validation
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    if not re.match(email_pattern, email):
        raise ValidationError(
            f"Invalid email format: {email}"
        )
        
    return email


def validate_dict_schema(data: Dict, 
                       schema: Dict[str, Dict[str, Any]]) -> Dict:
    """
    Validate a dictionary against a schema.
    
    Args:
        data: Dictionary to validate
        schema: Schema definition
            {
                'field_name': {
                    'type': type,
                    'required': bool,
                    'validator': callable (optional)
                }
            }
            
    Returns:
        Validated dictionary
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(data, dict):
        raise ValidationError(
            f"Expected dictionary, got {type(data).__name__}"
        )
        
    validated_data = {}
    
    # Track missing required fields
    missing_fields = []
    
    # Check required fields and apply validators
    for field_name, field_schema in schema.items():
        field_type = field_schema.get('type')
        required = field_schema.get('required', False)
        validator = field_schema.get('validator')
        
        if field_name in data:
            value = data[field_name]
            
            # Check type
            if field_type and not isinstance(value, field_type):
                raise ValidationError(
                    f"Invalid type for field '{field_name}': "
                    f"Expected {field_type.__name__}, got {type(value).__name__}"
                )
                
            # Apply validator if provided
            if validator and callable(validator):
                try:
                    value = validator(value)
                except ValidationError:
                    # Re-raise validation errors
                    raise
                except Exception as e:
                    raise ValidationError(
                        f"Validation failed for field '{field_name}': {e}"
                    )
                    
            # Add to validated data
            validated_data[field_name] = value
        elif required:
            missing_fields.append(field_name)
            
    if missing_fields:
        raise ValidationError(
            f"Missing required fields: {', '.join(missing_fields)}"
        )
        
    return validated_data


def validate_input(func=None, **validators):
    """
    Decorator to validate function arguments using provided validators.
    
    Usage:
        @validate_input(
            path=lambda x: validate_file_path(x, 'pdf'),
            value=lambda x: validate_numeric_input(x, min_value=0)
        )
        def process_file(path, value):
            # Function implementation
            
    Args:
        func: Function to decorate
        **validators: Mapping of parameter names to validator functions
        
    Returns:
        Decorated function
    """
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            # Get function signature
            import inspect
            sig = inspect.signature(f)
            bound_args = sig.bind(*args, **kwargs)
            
            # Apply validators to arguments
            for param_name, validator in validators.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    try:
                        bound_args.arguments[param_name] = validator(value)
                    except ValidationError as e:
                        # Add parameter name to error message
                        raise ValidationError(
                            f"Invalid value for '{param_name}': {e}"
                        )
                        
            # Call the original function with validated arguments
            return f(*bound_args.args, **bound_args.kwargs)
            
        return wrapper
        
    # Handle direct decoration without arguments
    if func:
        return decorator(func)
    return decorator