"""
Configuration module for the paper revision tool.

This module provides centralized configuration management for the tool,
including loading, validation, and accessing of configuration settings.
"""

from src.config.configuration import (
    AppConfig, LLMConfig, BudgetConfig, FileConfig,
    OutputConfig, LoggingConfig, APIConfig, OperationModeConfig,
    load_config, get_config, initialize_logging
)

from src.config.validation import (
    validate_config, validate_file_exists, validate_directory_exists,
    validate_api_key, validate_llm_model, validate_budget_settings,
    validate_api_config, verify_model_compatibility, test_api_connections
)

from src.config.environment import (
    load_env_files, validate_required_env_vars, get_env_var,
    get_api_key, initialize_environment
)

from src.config.cli import (
    create_argument_parser, parse_args, process_args,
    apply_args_to_config, validate_args, run_with_args
)

from src.config.schema import (
    validate_app_config_schema, generate_config_schema,
    save_schema_to_file, generate_sample_config, save_sample_config
)

from src.config.exceptions import (
    ConfigError, ConfigValidationError, ConfigFileError,
    EnvVarError, APIKeyError, ModelError, BudgetError,
    MissingRequiredArgumentError, InvalidArgumentError
)

# Version of the configuration module
__version__ = "1.0.0"