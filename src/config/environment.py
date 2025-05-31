"""
Environment variable management.

This module provides functions for loading and managing environment variables,
including loading from .env files, validating required variables, and masking
sensitive values for logging.
"""

import os
import logging
import re
from typing import Dict, List, Set, Optional, Any
from pathlib import Path
from dotenv import load_dotenv

# Configure logging for the environment module
logger = logging.getLogger(__name__)


def load_env_files(env_files: List[str] = None) -> None:
    """
    Load environment variables from .env files.
    
    Args:
        env_files: List of .env file paths to load. If not provided,
                  looks for .env files in standard locations.
    """
    if env_files is None:
        # Default locations to check for .env files
        env_files = [
            ".env",
            ".env.local",
            os.path.expanduser("~/.paperrevision/.env"),
        ]
    
    # Track which files were loaded
    loaded_files = []
    
    for env_file in env_files:
        if os.path.isfile(env_file):
            load_dotenv(env_file)
            loaded_files.append(env_file)
            logger.debug(f"Loaded environment variables from {env_file}")
    
    if loaded_files:
        logger.info(f"Loaded environment variables from: {', '.join(loaded_files)}")
    else:
        logger.warning("No .env files found. Using existing environment variables only.")


def validate_required_env_vars(required_vars: List[str]) -> List[str]:
    """
    Validate that required environment variables are set.
    
    Args:
        required_vars: List of required environment variable names
        
    Returns:
        List of missing environment variables
    """
    missing_vars = []
    
    for var in required_vars:
        if var not in os.environ or not os.environ[var]:
            missing_vars.append(var)
    
    if missing_vars:
        logger.warning(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    return missing_vars


def get_env_var(name: str, default: Any = None, required: bool = False) -> Any:
    """
    Get an environment variable with proper error handling.
    
    Args:
        name: Name of the environment variable
        default: Default value to return if the variable is not set
        required: Whether the variable is required
        
    Returns:
        The environment variable value or the default value
        
    Raises:
        ValueError: If the variable is required but not set
    """
    value = os.environ.get(name)
    
    if value is None or value == "":
        if required:
            raise ValueError(f"Required environment variable {name} is not set")
        return default
    
    return value


def get_api_key(provider: str) -> Optional[str]:
    """
    Get an API key for a provider from environment variables.
    
    Args:
        provider: Provider name (anthropic, openai, google, scopus, wos)
        
    Returns:
        API key if found, None otherwise
    """
    env_var_map = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "google": "GOOGLE_API_KEY",
        "scopus": "SCOPUS_API_KEY",
        "wos_username": "WOS_USERNAME",
        "wos_password": "WOS_PASSWORD",
    }
    
    env_var = env_var_map.get(provider.lower())
    if not env_var:
        logger.warning(f"Unknown provider: {provider}")
        return None
    
    api_key = os.environ.get(env_var)
    if not api_key:
        logger.warning(f"API key for {provider} not found in environment variables")
        return None
    
    return api_key


def mask_sensitive_value(value: str) -> str:
    """
    Mask sensitive values for logging.
    
    Args:
        value: Value to mask
        
    Returns:
        Masked value
    """
    if not value:
        return ""
    
    # Keep first 4 and last 4 characters, mask the rest
    if len(value) <= 8:
        return "*" * len(value)
    
    return f"{value[:4]}{'*' * (len(value) - 8)}{value[-4:]}"


def get_env_vars_summary(include_patterns: Optional[List[str]] = None,
                         exclude_patterns: Optional[List[str]] = None) -> Dict[str, str]:
    """
    Get a summary of environment variables for logging.
    
    Args:
        include_patterns: List of regex patterns to include
        exclude_patterns: List of regex patterns to exclude
        
    Returns:
        Dictionary of environment variables and their values (with sensitive values masked)
    """
    if include_patterns is None:
        include_patterns = [
            r"PAPERREVISION_.*",
            r".*_API_KEY",
            r"ANTHROPIC_.*",
            r"OPENAI_.*",
            r"GOOGLE_.*",
            r"SCOPUS_.*",
            r"WOS_.*",
        ]
    
    if exclude_patterns is None:
        exclude_patterns = [
            r".*_TOKEN",
            r".*_SECRET",
            r".*PASSWORD.*",
        ]
    
    # Compile patterns
    include_regexes = [re.compile(pattern) for pattern in include_patterns]
    exclude_regexes = [re.compile(pattern) for pattern in exclude_patterns]
    
    # Sensitive variable patterns for masking
    sensitive_patterns = [
        r".*_API_KEY",
        r".*_TOKEN",
        r".*_SECRET",
        r".*PASSWORD.*",
    ]
    sensitive_regexes = [re.compile(pattern) for pattern in sensitive_patterns]
    
    # Get matching variables
    result = {}
    for name, value in os.environ.items():
        # Check if the variable should be included
        include_match = any(regex.match(name) for regex in include_regexes)
        if not include_match:
            continue
        
        # Check if the variable should be excluded
        exclude_match = any(regex.match(name) for regex in exclude_regexes)
        if exclude_match:
            continue
        
        # Mask sensitive values
        is_sensitive = any(regex.match(name) for regex in sensitive_regexes)
        if is_sensitive:
            value = mask_sensitive_value(value)
        
        result[name] = value
    
    return result


def initialize_environment(env_files: List[str] = None) -> None:
    """
    Initialize environment variables.
    
    Args:
        env_files: List of .env file paths to load
    """
    # Load environment variables from .env files
    load_env_files(env_files)
    
    # Validate required environment variables for LLM APIs
    required_vars = []
    
    # Determine which API is being used based on provider environment variable
    provider = os.environ.get("PAPERREVISION_LLM_PROVIDER", "anthropic").lower()
    
    if provider == "anthropic":
        required_vars.append("ANTHROPIC_API_KEY")
    elif provider == "openai":
        required_vars.append("OPENAI_API_KEY")
    elif provider == "google":
        required_vars.append("GOOGLE_API_KEY")
    
    # Check for external APIs if enabled
    enabled_apis = os.environ.get("PAPERREVISION_API_ENABLED_APIS", "").lower()
    
    if "scopus" in enabled_apis:
        required_vars.append("SCOPUS_API_KEY")
    
    if "wos" in enabled_apis:
        required_vars.extend(["WOS_USERNAME", "WOS_PASSWORD"])
    
    # Validate required variables
    missing_vars = validate_required_env_vars(required_vars)
    
    if missing_vars:
        logger.warning(f"Missing required environment variables for {provider.upper()} API: {', '.join(missing_vars)}")
        logger.warning("You will need to provide these values when initializing the application.")
    else:
        logger.info(f"Environment variables for {provider.upper()} API validated successfully.")
    
    # Log environment variable summary
    env_vars_summary = get_env_vars_summary()
    logger.debug(f"Environment variables summary: {env_vars_summary}")


# Initialize environment variables on module import
if os.environ.get("PAPERREVISION_AUTO_LOAD_ENV", "true").lower() in {"true", "yes", "1", "on"}:
    initialize_environment()