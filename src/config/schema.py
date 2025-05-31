"""
Configuration schema and validation.

This module provides schema definitions for configuration validation,
including JSON schema generation and validation.
"""

import json
import logging
import jsonschema
from typing import Dict, Any, Optional, List, Set, Union, Callable
from dataclasses import asdict

from src.config.configuration import AppConfig, LLMConfig, BudgetConfig
from src.config.configuration import FileConfig, APIConfig, OutputConfig, LoggingConfig

# Configure logging for the schema module
logger = logging.getLogger(__name__)


# JSON Schema for LLM configuration
LLM_CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "provider": {
            "type": "string",
            "enum": ["anthropic", "openai", "google"],
            "description": "LLM provider name"
        },
        "model_name": {
            "type": "string",
            "description": "Model name for the selected provider"
        },
        "verify_model": {
            "type": "boolean",
            "description": "Whether to verify model availability on initialization"
        },
        "system_prompts": {
            "type": "object",
            "additionalProperties": {
                "type": "string"
            },
            "description": "System prompts for different tasks"
        },
        "competitor_evaluation": {
            "type": "boolean",
            "description": "Whether to use a competing model for evaluation"
        },
        "competing_evaluator": {
            "type": ["string", "null"],
            "description": "Competing model to use for evaluation (provider/model)"
        }
    },
    "required": ["provider", "model_name"],
    "additionalProperties": False
}

# JSON Schema for budget configuration
BUDGET_CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "budget": {
            "type": "number",
            "minimum": 0.01,
            "description": "Maximum budget in dollars"
        },
        "optimize_costs": {
            "type": "boolean",
            "description": "Whether to optimize for lower costs"
        },
        "warning_threshold": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "Threshold for budget warning (fraction of budget)"
        },
        "critical_threshold": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "Threshold for critical budget warning (fraction of budget)"
        }
    },
    "required": ["budget"],
    "additionalProperties": False
}

# JSON Schema for file configuration
FILE_CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "original_paper_path": {
            "type": "string",
            "description": "Path to the original paper PDF"
        },
        "reviewer_comment_files": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "description": "List of paths to reviewer comment files"
        },
        "editor_letter_path": {
            "type": ["string", "null"],
            "description": "Path to the editor letter file"
        },
        "output_dir": {
            "type": ["string", "null"],
            "description": "Directory for output files"
        },
        "use_cache": {
            "type": "boolean",
            "description": "Whether to use caching for API calls"
        }
    },
    "required": ["original_paper_path", "reviewer_comment_files"],
    "additionalProperties": False
}

# JSON Schema for output configuration
OUTPUT_CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "file_naming": {
            "type": "object",
            "additionalProperties": {
                "type": "string"
            },
            "description": "File naming templates for different output types"
        },
        "include_timestamp": {
            "type": "boolean",
            "description": "Whether to include timestamp in output file names"
        },
        "format": {
            "type": "string",
            "enum": ["docx", "pdf", "txt", "md"],
            "description": "Default output format"
        }
    },
    "additionalProperties": False
}

# JSON Schema for logging configuration
LOGGING_CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "level": {
            "type": "string",
            "enum": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            "description": "Logging level"
        },
        "log_to_file": {
            "type": "boolean",
            "description": "Whether to log to a file"
        },
        "log_dir": {
            "type": ["string", "null"],
            "description": "Directory for log files"
        },
        "log_format": {
            "type": "string",
            "description": "Format string for log messages"
        }
    },
    "additionalProperties": False
}

# JSON Schema for API configuration
API_CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "enabled_apis": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": ["scopus", "wos"]
            },
            "description": "List of enabled external APIs"
        },
        "scopus_api_key": {
            "type": ["string", "null"],
            "description": "API key for Scopus"
        },
        "wos_username": {
            "type": ["string", "null"],
            "description": "Username for Web of Science"
        },
        "wos_password": {
            "type": ["string", "null"],
            "description": "Password for Web of Science"
        }
    },
    "additionalProperties": False
}

# JSON Schema for operation mode configuration
OPERATION_MODE_CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "current_mode": {
            "type": "string",
            "description": "Current operation mode"
        },
        "modes": {
            "type": "object",
            "additionalProperties": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string"
                    },
                    "provider_recommendations": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "string"
                        }
                    },
                    "optimize_costs": {
                        "type": "boolean"
                    },
                    "budget": {
                        "type": "number"
                    },
                    "competitor_evaluation": {
                        "type": "boolean"
                    }
                }
            },
            "description": "Available operation modes"
        }
    },
    "required": ["current_mode"],
    "additionalProperties": False
}

# JSON Schema for complete application configuration
APP_CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "llm": LLM_CONFIG_SCHEMA,
        "budget": BUDGET_CONFIG_SCHEMA,
        "files": FILE_CONFIG_SCHEMA,
        "output": OUTPUT_CONFIG_SCHEMA,
        "logging": LOGGING_CONFIG_SCHEMA,
        "api": API_CONFIG_SCHEMA,
        "operation_mode": OPERATION_MODE_CONFIG_SCHEMA,
        "interactive": {
            "type": "boolean",
            "description": "Whether to run in interactive mode"
        },
        "run_id": {
            "type": ["string", "null"],
            "description": "Run ID for tracking"
        },
        "version": {
            "type": "string",
            "description": "Version of the application"
        }
    },
    "required": ["llm", "budget", "files"],
    "additionalProperties": False
}


def validate_config_against_schema(config: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
    """
    Validate configuration against a JSON schema.
    
    Args:
        config: Configuration dictionary to validate
        schema: JSON schema to validate against
        
    Returns:
        List of validation error messages
    """
    errors = []
    
    try:
        jsonschema.validate(config, schema)
    except jsonschema.exceptions.ValidationError as e:
        errors.append(f"Schema validation error: {e.message}")
    except Exception as e:
        errors.append(f"Error validating schema: {e}")
    
    return errors


def validate_app_config_schema(config: AppConfig) -> List[str]:
    """
    Validate the application configuration against its schema.
    
    Args:
        config: Application configuration to validate
        
    Returns:
        List of validation error messages
    """
    # Convert configuration to dictionary
    config_dict = asdict(config)
    
    # Validate against schema
    return validate_config_against_schema(config_dict, APP_CONFIG_SCHEMA)


def generate_config_schema() -> Dict[str, Any]:
    """
    Generate a JSON schema for the application configuration.
    
    Returns:
        JSON schema dictionary
    """
    schema = APP_CONFIG_SCHEMA.copy()
    schema["$schema"] = "http://json-schema.org/draft-07/schema#"
    schema["title"] = "Paper Revision Tool Configuration"
    schema["description"] = "Configuration schema for the Paper Revision Tool"
    
    return schema


def save_schema_to_file(file_path: str) -> None:
    """
    Save the configuration schema to a JSON file.
    
    Args:
        file_path: Path to save the schema to
    """
    schema = generate_config_schema()
    
    with open(file_path, "w") as f:
        json.dump(schema, f, indent=2)
    
    logger.info(f"Saved configuration schema to {file_path}")


def generate_sample_config() -> Dict[str, Any]:
    """
    Generate a sample configuration.
    
    Returns:
        Sample configuration dictionary
    """
    # Create a default configuration
    config = AppConfig()
    
    # Convert to dictionary
    config_dict = asdict(config)
    
    return config_dict


def save_sample_config(file_path: str, format: str = "json") -> None:
    """
    Save a sample configuration to a file.
    
    Args:
        file_path: Path to save the configuration to
        format: Format to save the configuration in (json or yaml)
    """
    config = generate_sample_config()
    
    if format.lower() == "json":
        with open(file_path, "w") as f:
            json.dump(config, f, indent=2)
    elif format.lower() in ["yaml", "yml"]:
        import yaml
        with open(file_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Saved sample configuration to {file_path}")