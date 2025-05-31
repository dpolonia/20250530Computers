"""
Configuration validation module.

This module provides functions for validating configuration settings,
including file paths, API keys, and model availability.
"""

import os
import logging
import json
import yaml
from typing import Dict, Any, Optional, List, Set, Union, Callable, Tuple, Type
from pathlib import Path

from src.config.configuration import AppConfig, LLMConfig, BudgetConfig, APIConfig

# Configure logging for the validation module
logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Exception raised for configuration validation errors."""
    pass


def validate_file_exists(file_path: str, description: str) -> bool:
    """
    Validate that a file exists.
    
    Args:
        file_path: Path to the file
        description: Description of the file for error messages
        
    Returns:
        True if the file exists, False otherwise
    """
    if not file_path:
        logger.warning(f"No {description} file path provided.")
        return False
    
    if not os.path.exists(file_path):
        logger.warning(f"{description} file not found: {file_path}")
        return False
    
    return True


def validate_directory_exists(dir_path: str, description: str, create: bool = False) -> bool:
    """
    Validate that a directory exists.
    
    Args:
        dir_path: Path to the directory
        description: Description of the directory for error messages
        create: Whether to create the directory if it doesn't exist
        
    Returns:
        True if the directory exists or was created, False otherwise
    """
    if not dir_path:
        logger.warning(f"No {description} directory path provided.")
        return False
    
    if not os.path.exists(dir_path):
        if create:
            try:
                os.makedirs(dir_path, exist_ok=True)
                logger.info(f"Created {description} directory: {dir_path}")
                return True
            except Exception as e:
                logger.warning(f"Failed to create {description} directory: {e}")
                return False
        else:
            logger.warning(f"{description} directory not found: {dir_path}")
            return False
    
    if not os.path.isdir(dir_path):
        logger.warning(f"{description} path is not a directory: {dir_path}")
        return False
    
    return True


def validate_api_key(api_key: Optional[str], provider: str) -> bool:
    """
    Validate that an API key is provided.
    
    Args:
        api_key: API key to validate
        provider: Provider name for error messages
        
    Returns:
        True if the API key is provided, False otherwise
    """
    if not api_key:
        logger.warning(f"No API key provided for {provider}.")
        return False
    
    # Basic validation of key format
    if len(api_key) < 10:
        logger.warning(f"API key for {provider} is too short. This may not be a valid key.")
        return False
    
    return True


def validate_llm_model(provider: str, model_name: str) -> Tuple[bool, Optional[str]]:
    """
    Validate that a model is available for the given provider.
    
    Args:
        provider: Provider name
        model_name: Model name
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Import model validation functions based on provider
    if provider == "anthropic":
        try:
            from src.models.anthropic_models import get_claude_model_info, list_available_models
            model_info = get_claude_model_info(model_name)
            if not model_info:
                available_models = list_available_models()
                return False, f"Model '{model_name}' not found for provider '{provider}'. Available models: {', '.join(available_models)}"
            return True, None
        except ImportError:
            return False, f"Could not import Anthropic model validation. Is the anthropic package installed?"
    
    elif provider == "openai":
        try:
            from src.models.openai_models import get_openai_model_info, list_available_models
            model_info = get_openai_model_info(model_name)
            if not model_info:
                available_models = list_available_models()
                return False, f"Model '{model_name}' not found for provider '{provider}'. Available models: {', '.join(available_models)}"
            return True, None
        except ImportError:
            return False, f"Could not import OpenAI model validation. Is the openai package installed?"
    
    elif provider == "google":
        try:
            from src.models.google_models import get_gemini_model_info, list_available_models
            model_info = get_gemini_model_info(model_name)
            if not model_info:
                available_models = list_available_models()
                return False, f"Model '{model_name}' not found for provider '{provider}'. Available models: {', '.join(available_models)}"
            return True, None
        except ImportError:
            return False, f"Could not import Google model validation. Is the google-generativeai package installed?"
    
    else:
        return False, f"Unsupported provider: {provider}"


def validate_budget_settings(budget: BudgetConfig) -> Tuple[bool, Optional[str]]:
    """
    Validate budget configuration settings.
    
    Args:
        budget: Budget configuration
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if budget.budget <= 0:
        return False, f"Budget must be greater than 0. Got: {budget.budget}"
    
    if not 0 < budget.warning_threshold < 1:
        return False, f"Warning threshold must be between 0 and 1. Got: {budget.warning_threshold}"
    
    if not 0 < budget.critical_threshold < 1:
        return False, f"Critical threshold must be between 0 and 1. Got: {budget.critical_threshold}"
    
    if budget.critical_threshold <= budget.warning_threshold:
        return False, f"Critical threshold ({budget.critical_threshold}) must be greater than warning threshold ({budget.warning_threshold})"
    
    return True, None


def validate_api_config(api_config: APIConfig) -> List[str]:
    """
    Validate API configuration and return list of warnings.
    
    Args:
        api_config: API configuration
        
    Returns:
        List of warning messages
    """
    warnings = []
    
    # Validate Scopus API
    if "scopus" in api_config.enabled_apis:
        if not api_config.scopus_api_key:
            warnings.append("Scopus API enabled but no API key provided.")
        elif len(api_config.scopus_api_key) < 10:
            warnings.append("Scopus API key is too short. This may not be a valid key.")
    
    # Validate WoS API
    if "wos" in api_config.enabled_apis:
        if not api_config.wos_username:
            warnings.append("Web of Science API enabled but no username provided.")
        if not api_config.wos_password:
            warnings.append("Web of Science API enabled but no password provided.")
    
    return warnings


def validate_config(config: AppConfig) -> List[str]:
    """
    Validate the configuration and return a list of warnings.
    
    Args:
        config: Configuration to validate
        
    Returns:
        List of warning messages
    """
    warnings = []
    
    # Validate file paths
    if not validate_file_exists(config.files.original_paper_path, "original paper"):
        warnings.append(f"Original paper file not found: {config.files.original_paper_path}")
    
    if not config.files.reviewer_comment_files:
        warnings.append("No reviewer comment files provided.")
    else:
        for i, file_path in enumerate(config.files.reviewer_comment_files):
            if not validate_file_exists(file_path, f"reviewer {i+1} comment"):
                warnings.append(f"Reviewer {i+1} comment file not found: {file_path}")
    
    if config.files.editor_letter_path and not validate_file_exists(config.files.editor_letter_path, "editor letter"):
        warnings.append(f"Editor letter file not found: {config.files.editor_letter_path}")
    
    # Validate output directory
    output_dir = config.files.output_dir
    if output_dir and not validate_directory_exists(output_dir, "output", create=True):
        warnings.append(f"Failed to access or create output directory: {output_dir}")
    
    # Validate LLM model
    is_valid_model, model_error = validate_llm_model(config.llm.provider, config.llm.model_name)
    if not is_valid_model and model_error:
        warnings.append(model_error)
    
    # Validate budget settings
    is_valid_budget, budget_error = validate_budget_settings(config.budget)
    if not is_valid_budget and budget_error:
        warnings.append(budget_error)
    
    # Validate API configuration
    api_warnings = validate_api_config(config.api)
    warnings.extend(api_warnings)
    
    # Validate operation mode
    if config.operation_mode.current_mode not in config.operation_mode.modes:
        warnings.append(f"Invalid operation mode: {config.operation_mode.current_mode}")
    
    return warnings


def verify_model_compatibility(config: AppConfig) -> Tuple[bool, List[str]]:
    """
    Verify that the model is compatible with the configuration.
    
    Args:
        config: Configuration to verify
        
    Returns:
        Tuple of (is_compatible, warnings)
    """
    warnings = []
    
    # Get model information
    provider = config.llm.provider
    model_name = config.llm.model_name
    
    try:
        if provider == "anthropic":
            from src.models.anthropic_models import get_claude_model_info
            model_info = get_claude_model_info(model_name)
        elif provider == "openai":
            from src.models.openai_models import get_openai_model_info
            model_info = get_openai_model_info(model_name)
        elif provider == "google":
            from src.models.google_models import get_gemini_model_info
            model_info = get_gemini_model_info(model_name)
        else:
            warnings.append(f"Unsupported provider: {provider}")
            return False, warnings
        
        if not model_info:
            warnings.append(f"Model '{model_name}' not found for provider '{provider}'")
            return False, warnings
        
        # Check if the model is suitable for the operation mode
        operation_mode = config.operation_mode.current_mode
        mode_info = config.operation_mode.modes.get(operation_mode, {})
        recommended_model = mode_info.get("provider_recommendations", {}).get(provider)
        
        if recommended_model and model_name != recommended_model:
            warnings.append(f"Model '{model_name}' is not the recommended model for {operation_mode} mode. "
                           f"Recommended model: '{recommended_model}'")
        
        # Check token limits
        context_window = model_info.get("context_window", 0)
        output_tokens = model_info.get("output_tokens", 0)
        
        if context_window < 8000:
            warnings.append(f"Model '{model_name}' has a small context window ({context_window} tokens). "
                          f"This may limit analysis of large papers.")
        
        if output_tokens < 4000:
            warnings.append(f"Model '{model_name}' has a small output token limit ({output_tokens} tokens). "
                          f"This may limit generation of detailed solutions.")
        
        # Check pricing for budget compatibility
        price_per_1k_input = model_info.get("price_per_1k_input", 0)
        price_per_1k_output = model_info.get("price_per_1k_output", 0)
        
        estimated_token_usage = 100000  # Rough estimate for a typical paper revision
        estimated_cost = (estimated_token_usage / 1000) * (price_per_1k_input + price_per_1k_output) / 2
        
        if estimated_cost > config.budget.budget:
            warnings.append(f"Estimated cost for a typical paper revision with model '{model_name}' "
                          f"(${estimated_cost:.2f}) exceeds the budget (${config.budget.budget:.2f}).")
        
        return True, warnings
        
    except ImportError as e:
        warnings.append(f"Could not import model information: {e}")
        return False, warnings


def test_api_connections(config: AppConfig) -> Dict[str, bool]:
    """
    Test connections to enabled APIs.
    
    Args:
        config: Configuration with API settings
        
    Returns:
        Dictionary mapping API names to connection status
    """
    results = {}
    
    # Test LLM API
    provider = config.llm.provider
    model_name = config.llm.model_name
    
    try:
        from src.utils.llm_client import get_llm_client
        llm_client = get_llm_client(provider, model_name, verify=False)
        results["llm"] = llm_client.validate_api_key()
    except Exception as e:
        logger.warning(f"Failed to test LLM API connection: {e}")
        results["llm"] = False
    
    # Test Scopus API
    if "scopus" in config.api.enabled_apis:
        try:
            from src.utils.scopus_client import ScopusClient
            scopus_client = ScopusClient(config.api.scopus_api_key)
            results["scopus"] = scopus_client.test_connection()
        except Exception as e:
            logger.warning(f"Failed to test Scopus API connection: {e}")
            results["scopus"] = False
    
    # Test WoS API
    if "wos" in config.api.enabled_apis:
        try:
            from src.utils.wos_client import WoSClient
            wos_client = WoSClient(config.api.wos_username, config.api.wos_password)
            results["wos"] = wos_client.test_connection()
        except Exception as e:
            logger.warning(f"Failed to test Web of Science API connection: {e}")
            results["wos"] = False
    
    return results