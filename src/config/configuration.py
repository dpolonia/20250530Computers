"""
Configuration system for the paper revision tool.

This module provides a centralized, validated configuration system with support for:
1. Environment variables
2. Configuration files
3. Command line arguments
4. Default values
5. Validation of settings
"""

import os
import json
import logging
import yaml
from typing import Dict, Any, Optional, List, Set, Union, Callable
from pathlib import Path
from dataclasses import dataclass, field, asdict

# Configure logging for the configuration module
logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for LLM providers and models."""
    
    provider: str = "anthropic"
    model_name: str = "claude-3-opus-20240229"
    verify_model: bool = True
    system_prompts: Dict[str, str] = field(default_factory=dict)
    competitor_evaluation: bool = False
    competing_evaluator: Optional[str] = None
    
    def __post_init__(self):
        """Validate LLM configuration after initialization."""
        # Normalize provider name
        self.provider = self.provider.lower()
        
        # Validate provider
        valid_providers = {"anthropic", "openai", "google"}
        if self.provider not in valid_providers:
            logger.warning(f"Invalid provider: {self.provider}. Using default (anthropic).")
            self.provider = "anthropic"
        
        # Ensure system prompts has default values if not provided
        if not self.system_prompts:
            self.system_prompts = {
                "paper_analysis": "You are a scientific paper analysis assistant. Analyze the paper thoroughly and extract key information.",
                "reviewer_analysis": "You are a scientific reviewer analysis assistant. Extract key insights from reviewer comments.",
                "solution_generation": "You are a scientific paper revision assistant. Generate effective solutions to address reviewer concerns.",
                "changes_generation": "You are a scientific paper revision assistant. Generate specific text changes to implement revision solutions.",
                "reference_validation": "You are a scientific reference assistant. Validate and suggest improvements to references.",
                "assessment": "You are a scientific paper assessment assistant. Evaluate the impact of revisions on paper quality.",
                "editor_letter": "You are a scientific communication assistant. Create professional response letters to journal editors."
            }


@dataclass
class BudgetConfig:
    """Configuration for budget and cost management."""
    
    budget: float = 10.0
    optimize_costs: bool = False
    warning_threshold: float = 0.75
    critical_threshold: float = 0.90
    
    def __post_init__(self):
        """Validate budget configuration after initialization."""
        # Ensure budget is positive
        if self.budget <= 0:
            logger.warning(f"Invalid budget: {self.budget}. Using default (10.0).")
            self.budget = 10.0
        
        # Ensure thresholds are between 0 and 1
        if not 0 < self.warning_threshold < 1:
            logger.warning(f"Invalid warning threshold: {self.warning_threshold}. Using default (0.75).")
            self.warning_threshold = 0.75
        
        if not 0 < self.critical_threshold < 1:
            logger.warning(f"Invalid critical threshold: {self.critical_threshold}. Using default (0.90).")
            self.critical_threshold = 0.90
        
        # Ensure critical threshold is greater than warning threshold
        if self.critical_threshold <= self.warning_threshold:
            logger.warning(f"Critical threshold ({self.critical_threshold}) must be greater than warning threshold ({self.warning_threshold}). Adjusting to 0.90.")
            self.critical_threshold = 0.90


@dataclass
class FileConfig:
    """Configuration for file paths and output settings."""
    
    original_paper_path: str = ""
    reviewer_comment_files: List[str] = field(default_factory=list)
    editor_letter_path: Optional[str] = None
    output_dir: Optional[str] = None
    use_cache: bool = True
    
    def __post_init__(self):
        """Validate file configuration after initialization."""
        # Validate original paper path
        if self.original_paper_path and not os.path.exists(self.original_paper_path):
            logger.warning(f"Original paper file not found: {self.original_paper_path}")
        
        # Validate reviewer comment files
        valid_reviewer_files = []
        for file_path in self.reviewer_comment_files:
            if os.path.exists(file_path):
                valid_reviewer_files.append(file_path)
            else:
                logger.warning(f"Reviewer comment file not found: {file_path}")
        self.reviewer_comment_files = valid_reviewer_files
        
        # Validate editor letter path
        if self.editor_letter_path and not os.path.exists(self.editor_letter_path):
            logger.warning(f"Editor letter file not found: {self.editor_letter_path}")
            self.editor_letter_path = None
        
        # Create output directory if it doesn't exist
        if self.output_dir and not os.path.exists(self.output_dir):
            logger.info(f"Creating output directory: {self.output_dir}")
            os.makedirs(self.output_dir, exist_ok=True)


@dataclass
class OutputConfig:
    """Configuration for output file naming and formats."""
    
    file_naming: Dict[str, str] = field(default_factory=dict)
    include_timestamp: bool = True
    format: str = "docx"
    
    def __post_init__(self):
        """Initialize with default values if not provided."""
        if not self.file_naming:
            self.file_naming = {
                "analysis": "paper_analysis.json",
                "reviewer_analysis": "reviewer_analysis.json",
                "changes_document": "changes_document.docx",
                "revised_paper": "revised_paper.docx",
                "assessment": "assessment.docx",
                "editor_letter": "editor_letter.docx",
                "references": "references.bib",
                "cost_report": "cost_report.txt"
            }
        
        # Validate format
        valid_formats = {"docx", "pdf", "txt", "md"}
        if self.format not in valid_formats:
            logger.warning(f"Invalid output format: {self.format}. Using default (docx).")
            self.format = "docx"


@dataclass
class LoggingConfig:
    """Configuration for logging settings."""
    
    level: str = "INFO"
    log_to_file: bool = True
    log_dir: Optional[str] = None
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    def __post_init__(self):
        """Validate logging configuration after initialization."""
        # Normalize log level
        self.level = self.level.upper()
        
        # Validate log level
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.level not in valid_levels:
            logger.warning(f"Invalid log level: {self.level}. Using default (INFO).")
            self.level = "INFO"
        
        # Create log directory if it doesn't exist
        if self.log_to_file and self.log_dir and not os.path.exists(self.log_dir):
            logger.info(f"Creating log directory: {self.log_dir}")
            os.makedirs(self.log_dir, exist_ok=True)


@dataclass
class APIConfig:
    """Configuration for external API integration."""
    
    enabled_apis: List[str] = field(default_factory=list)
    scopus_api_key: Optional[str] = None
    wos_username: Optional[str] = None
    wos_password: Optional[str] = None
    
    def __post_init__(self):
        """Validate API configuration after initialization."""
        # Normalize API names
        self.enabled_apis = [api.lower() for api in self.enabled_apis]
        
        # Validate APIs
        valid_apis = {"scopus", "wos"}
        invalid_apis = [api for api in self.enabled_apis if api not in valid_apis]
        if invalid_apis:
            logger.warning(f"Invalid APIs: {', '.join(invalid_apis)}. They will be ignored.")
            self.enabled_apis = [api for api in self.enabled_apis if api in valid_apis]
        
        # Check if required API keys are available
        if "scopus" in self.enabled_apis and not self.scopus_api_key:
            self.scopus_api_key = os.environ.get("SCOPUS_API_KEY")
            if not self.scopus_api_key:
                logger.warning("Scopus API enabled but no API key provided. Scopus integration will be disabled.")
                self.enabled_apis.remove("scopus")
        
        if "wos" in self.enabled_apis and (not self.wos_username or not self.wos_password):
            self.wos_username = os.environ.get("WOS_USERNAME")
            self.wos_password = os.environ.get("WOS_PASSWORD")
            if not self.wos_username or not self.wos_password:
                logger.warning("Web of Science API enabled but no credentials provided. WoS integration will be disabled.")
                self.enabled_apis.remove("wos")


@dataclass
class OperationModeConfig:
    """Configuration for operation modes (training, finetuning, final)."""
    
    current_mode: str = "finetuning"
    modes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize with default values if not provided."""
        if not self.modes:
            self.modes = {
                "training": {
                    "description": "Training Mode - Less expensive models, optimized for cost",
                    "provider_recommendations": {
                        "anthropic": "claude-3-haiku-20240307",
                        "openai": "gpt-4o-mini",
                        "google": "gemini-1.5-flash"
                    },
                    "optimize_costs": True,
                    "budget": 5.0,
                    "competitor_evaluation": False
                },
                "finetuning": {
                    "description": "Fine-tuning Mode - Balanced models with moderate quality",
                    "provider_recommendations": {
                        "anthropic": "claude-3-5-sonnet-20241022",
                        "openai": "gpt-4o",
                        "google": "gemini-1.5-pro"
                    },
                    "optimize_costs": True,
                    "budget": 10.0,
                    "competitor_evaluation": True
                },
                "final": {
                    "description": "Final Mode - Highest quality models, best output quality",
                    "provider_recommendations": {
                        "anthropic": "claude-opus-4-20250514",
                        "openai": "gpt-4.5-preview",
                        "google": "gemini-2.5-pro-preview"
                    },
                    "optimize_costs": False,
                    "budget": 20.0,
                    "competitor_evaluation": True
                }
            }
        
        # Validate current mode
        if self.current_mode not in self.modes:
            logger.warning(f"Invalid operation mode: {self.current_mode}. Using default (finetuning).")
            self.current_mode = "finetuning"


@dataclass
class AppConfig:
    """
    Main configuration class that contains all configuration components.
    
    This class serves as the main entry point for accessing configuration
    settings across the application.
    """
    
    llm: LLMConfig = field(default_factory=LLMConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    files: FileConfig = field(default_factory=FileConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    api: APIConfig = field(default_factory=APIConfig)
    operation_mode: OperationModeConfig = field(default_factory=OperationModeConfig)
    
    # Application-wide settings
    interactive: bool = False
    run_id: Optional[str] = None
    version: str = "1.0.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert configuration to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    def save(self, file_path: str) -> None:
        """Save configuration to file."""
        extension = os.path.splitext(file_path)[1].lower()
        
        if extension == ".json":
            with open(file_path, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
        elif extension in {".yml", ".yaml"}:
            with open(file_path, "w") as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported file format: {extension}")
    
    def apply_operation_mode(self) -> None:
        """Apply settings from the current operation mode."""
        mode_settings = self.operation_mode.modes.get(self.operation_mode.current_mode, {})
        
        # Apply mode-specific settings
        if "optimize_costs" in mode_settings:
            self.budget.optimize_costs = mode_settings["optimize_costs"]
        
        if "budget" in mode_settings:
            self.budget.budget = mode_settings["budget"]
        
        if "competitor_evaluation" in mode_settings:
            self.llm.competitor_evaluation = mode_settings["competitor_evaluation"]
        
        # Apply provider-specific model recommendation if available
        provider_recommendations = mode_settings.get("provider_recommendations", {})
        if self.llm.provider in provider_recommendations:
            self.llm.model_name = provider_recommendations[self.llm.provider]


class ConfigLoader:
    """
    Loads and validates configuration from multiple sources.
    
    The loader follows this precedence order (highest to lowest):
    1. Command line arguments
    2. Environment variables
    3. Configuration file
    4. Default values
    """
    
    def __init__(self):
        """Initialize the configuration loader."""
        self.config = AppConfig()
        self.env_prefix = "PAPERREVISION_"
    
    def load_defaults(self) -> None:
        """Load default configuration values."""
        # Default values are already set in the dataclass definitions
        pass
    
    def load_from_file(self, file_path: str) -> None:
        """
        Load configuration from a file.
        
        Args:
            file_path: Path to the configuration file
        """
        if not os.path.exists(file_path):
            logger.warning(f"Configuration file not found: {file_path}")
            return
        
        try:
            extension = os.path.splitext(file_path)[1].lower()
            
            if extension == ".json":
                with open(file_path, "r") as f:
                    config_data = json.load(f)
            elif extension in {".yml", ".yaml"}:
                with open(file_path, "r") as f:
                    config_data = yaml.safe_load(f)
            else:
                logger.warning(f"Unsupported configuration file format: {extension}")
                return
            
            self._update_config_from_dict(config_data)
            logger.info(f"Loaded configuration from {file_path}")
            
        except Exception as e:
            logger.warning(f"Error loading configuration from {file_path}: {e}")
    
    def load_from_env(self) -> None:
        """Load configuration from environment variables."""
        # Get all environment variables with the prefix
        env_vars = {k: v for k, v in os.environ.items() if k.startswith(self.env_prefix)}
        
        # Convert environment variables to a nested dictionary
        config_data = {}
        for key, value in env_vars.items():
            # Remove prefix and convert to lowercase
            key = key[len(self.env_prefix):].lower()
            
            # Split by double underscore to get nested keys
            parts = key.split("__")
            
            # Build nested dictionary
            current = config_data
            for i, part in enumerate(parts):
                if i == len(parts) - 1:
                    # Convert value to appropriate type
                    current[part] = self._convert_env_value(value)
                else:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
        
        # Update configuration with environment variables
        self._update_config_from_dict(config_data)
        
        # Special handling for API keys that don't use the prefix
        self._load_api_keys_from_env()
    
    def _load_api_keys_from_env(self) -> None:
        """Load API keys from environment variables without the prefix."""
        # Load Anthropic API key
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        if anthropic_key and self.config.llm.provider == "anthropic":
            logger.debug("Found Anthropic API key in environment")
        
        # Load OpenAI API key
        openai_key = os.environ.get("OPENAI_API_KEY")
        if openai_key and self.config.llm.provider == "openai":
            logger.debug("Found OpenAI API key in environment")
        
        # Load Google API key
        google_key = os.environ.get("GOOGLE_API_KEY")
        if google_key and self.config.llm.provider == "google":
            logger.debug("Found Google API key in environment")
        
        # Load Scopus API key
        scopus_key = os.environ.get("SCOPUS_API_KEY")
        if scopus_key and "scopus" in self.config.api.enabled_apis:
            self.config.api.scopus_api_key = scopus_key
            logger.debug("Found Scopus API key in environment")
        
        # Load WoS credentials
        wos_username = os.environ.get("WOS_USERNAME")
        wos_password = os.environ.get("WOS_PASSWORD")
        if wos_username and wos_password and "wos" in self.config.api.enabled_apis:
            self.config.api.wos_username = wos_username
            self.config.api.wos_password = wos_password
            logger.debug("Found Web of Science credentials in environment")
    
    def load_from_args(self, args: Dict[str, Any]) -> None:
        """
        Load configuration from command line arguments.
        
        Args:
            args: Dictionary of command line arguments
        """
        config_data = {}
        
        # Map command line arguments to configuration
        
        # LLM configuration
        if "provider" in args and args["provider"]:
            config_data["llm"] = config_data.get("llm", {})
            config_data["llm"]["provider"] = args["provider"]
        
        if "model" in args and args["model"]:
            config_data["llm"] = config_data.get("llm", {})
            config_data["llm"]["model_name"] = args["model"]
        
        # Budget configuration
        if "optimize_costs" in args:
            config_data["budget"] = config_data.get("budget", {})
            config_data["budget"]["optimize_costs"] = args["optimize_costs"]
        
        if "budget" in args and args["budget"] is not None:
            config_data["budget"] = config_data.get("budget", {})
            config_data["budget"]["budget"] = float(args["budget"])
        
        # File configuration
        if "paper" in args and args["paper"]:
            config_data["files"] = config_data.get("files", {})
            config_data["files"]["original_paper_path"] = args["paper"]
        
        if "reviewer_comments" in args and args["reviewer_comments"]:
            config_data["files"] = config_data.get("files", {})
            config_data["files"]["reviewer_comment_files"] = args["reviewer_comments"]
        
        if "editor_letter" in args and args["editor_letter"]:
            config_data["files"] = config_data.get("files", {})
            config_data["files"]["editor_letter_path"] = args["editor_letter"]
        
        if "output_dir" in args and args["output_dir"]:
            config_data["files"] = config_data.get("files", {})
            config_data["files"]["output_dir"] = args["output_dir"]
        
        # Logging configuration
        if "verbose" in args:
            config_data["logging"] = config_data.get("logging", {})
            config_data["logging"]["level"] = "DEBUG" if args["verbose"] else "INFO"
        
        # API configuration
        if "api" in args and args["api"]:
            config_data["api"] = config_data.get("api", {})
            config_data["api"]["enabled_apis"] = args["api"].split(",") if isinstance(args["api"], str) else args["api"]
        
        if "key" in args and args["key"]:
            config_data["api"] = config_data.get("api", {})
            config_data["api"]["scopus_api_key"] = args["key"]
        
        # Operation mode configuration
        if "mode" in args and args["mode"]:
            config_data["operation_mode"] = config_data.get("operation_mode", {})
            config_data["operation_mode"]["current_mode"] = args["mode"]
        
        # Application-wide settings
        if "interactive" in args:
            config_data["interactive"] = args["interactive"]
        
        if "run_id" in args and args["run_id"]:
            config_data["run_id"] = args["run_id"]
        
        # Update configuration with command line arguments
        self._update_config_from_dict(config_data)
    
    def _update_config_from_dict(self, data: Dict[str, Any]) -> None:
        """
        Update configuration from a dictionary.
        
        Args:
            data: Dictionary with configuration data
        """
        # Update LLM configuration
        if "llm" in data:
            for key, value in data["llm"].items():
                if hasattr(self.config.llm, key):
                    setattr(self.config.llm, key, value)
        
        # Update budget configuration
        if "budget" in data:
            for key, value in data["budget"].items():
                if hasattr(self.config.budget, key):
                    setattr(self.config.budget, key, value)
        
        # Update file configuration
        if "files" in data:
            for key, value in data["files"].items():
                if hasattr(self.config.files, key):
                    setattr(self.config.files, key, value)
        
        # Update output configuration
        if "output" in data:
            for key, value in data["output"].items():
                if hasattr(self.config.output, key):
                    setattr(self.config.output, key, value)
        
        # Update logging configuration
        if "logging" in data:
            for key, value in data["logging"].items():
                if hasattr(self.config.logging, key):
                    setattr(self.config.logging, key, value)
        
        # Update API configuration
        if "api" in data:
            for key, value in data["api"].items():
                if hasattr(self.config.api, key):
                    setattr(self.config.api, key, value)
        
        # Update operation mode configuration
        if "operation_mode" in data:
            if "current_mode" in data["operation_mode"]:
                self.config.operation_mode.current_mode = data["operation_mode"]["current_mode"]
            
            if "modes" in data["operation_mode"]:
                self.config.operation_mode.modes.update(data["operation_mode"]["modes"])
        
        # Update application-wide settings
        for key in ["interactive", "run_id", "version"]:
            if key in data:
                setattr(self.config, key, data[key])
    
    def _convert_env_value(self, value: str) -> Any:
        """
        Convert environment variable value to appropriate type.
        
        Args:
            value: String value from environment variable
            
        Returns:
            Converted value
        """
        # Convert to boolean
        if value.lower() in {"true", "yes", "1", "on"}:
            return True
        elif value.lower() in {"false", "no", "0", "off"}:
            return False
        
        # Convert to integer
        try:
            int_value = int(value)
            return int_value
        except ValueError:
            pass
        
        # Convert to float
        try:
            float_value = float(value)
            return float_value
        except ValueError:
            pass
        
        # Convert to list (comma-separated values)
        if "," in value:
            return [item.strip() for item in value.split(",")]
        
        # Return as string
        return value
    
    def validate(self) -> None:
        """Validate the loaded configuration."""
        # Validation is handled in the __post_init__ methods of each configuration class
        pass
    
    def finalize(self) -> None:
        """
        Finalize the configuration after loading.
        
        This applies settings from the current operation mode and
        performs any other necessary finalizations.
        """
        # Apply settings from the current operation mode
        self.config.apply_operation_mode()
    
    def load(self, config_file: Optional[str] = None, args: Optional[Dict[str, Any]] = None) -> AppConfig:
        """
        Load configuration from all sources.
        
        Args:
            config_file: Optional path to configuration file
            args: Optional dictionary of command line arguments
            
        Returns:
            The loaded configuration
        """
        # Load defaults
        self.load_defaults()
        
        # Load from configuration file if provided
        if config_file:
            self.load_from_file(config_file)
        
        # Load from environment variables
        self.load_from_env()
        
        # Load from command line arguments if provided
        if args:
            self.load_from_args(args)
        
        # Validate configuration
        self.validate()
        
        # Finalize configuration
        self.finalize()
        
        return self.config


# Global configuration instance
_config: Optional[AppConfig] = None


def load_config(config_file: Optional[str] = None, args: Optional[Dict[str, Any]] = None) -> AppConfig:
    """
    Load configuration from all sources.
    
    Args:
        config_file: Optional path to configuration file
        args: Optional dictionary of command line arguments
        
    Returns:
        The loaded configuration
    """
    global _config
    
    loader = ConfigLoader()
    _config = loader.load(config_file, args)
    
    return _config


def get_config() -> AppConfig:
    """
    Get the current configuration.
    
    Returns:
        The current configuration
        
    Raises:
        RuntimeError: If configuration has not been loaded
    """
    if _config is None:
        raise RuntimeError("Configuration has not been loaded. Call load_config() first.")
    
    return _config


def initialize_logging(config: Optional[AppConfig] = None) -> None:
    """
    Initialize logging based on configuration.
    
    Args:
        config: Optional configuration instance (uses global config if not provided)
    """
    if config is None:
        config = get_config()
    
    log_config = config.logging
    
    # Get numeric log level
    level = getattr(logging, log_config.level)
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format=log_config.log_format
    )
    
    # Configure file logging if enabled
    if log_config.log_to_file:
        log_dir = log_config.log_dir or (config.files.output_dir or "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        # Generate log file path
        log_file = os.path.join(log_dir, f"paper_revision_{config.run_id or datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(log_config.log_format))
        
        # Add handler to root logger
        logging.getLogger().addHandler(file_handler)
    
    # Log configuration summary
    logger.info(f"Logging initialized at level {log_config.level}")
    if log_config.log_to_file:
        logger.info(f"Logs will be written to {log_file}")


# Load default configuration on module import
load_config()