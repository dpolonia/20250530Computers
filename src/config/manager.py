"""
Configuration manager module.

This module provides a centralized configuration manager that handles
loading, validation, and accessing of configuration settings.
"""

import os
import logging
import json
import yaml
from typing import Dict, Any, List, Optional, Set, Union
import datetime

from src.config.configuration import (
    AppConfig, load_config, get_config, initialize_logging
)
from src.config.validation import (
    validate_config, validate_llm_model, test_api_connections
)
from src.config.environment import (
    load_env_files, get_env_var, initialize_environment
)
from src.config.cli import (
    parse_args, process_args, apply_args_to_config, validate_args
)
from src.config.schema import (
    validate_app_config_schema
)

# Configure logging for the manager module
logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Configuration manager that handles loading, validation, and accessing of
    configuration settings.
    """
    
    def __init__(self):
        """Initialize the configuration manager."""
        self.config = None
        self.args = None
        self.warnings = []
    
    def load_from_args(self, args: Optional[List[str]] = None) -> AppConfig:
        """
        Load configuration from command-line arguments.
        
        Args:
            args: Optional list of command-line arguments. If None, uses sys.argv.
            
        Returns:
            Loaded configuration
        """
        # Parse and process arguments
        parsed_args = parse_args(args)
        self.args = process_args(parsed_args)
        
        # Validate arguments
        arg_errors = validate_args(self.args)
        if arg_errors:
            for error in arg_errors:
                logger.error(f"Argument error: {error}")
            raise ValueError(f"Invalid arguments: {', '.join(arg_errors)}")
        
        # Load configuration from file and/or environment
        config_file = self.args.get("config")
        self.config = load_config(config_file, self.args)
        
        # Initialize logging
        initialize_logging(self.config)
        
        # Validate configuration
        self.warnings = validate_config(self.config)
        for warning in self.warnings:
            logger.warning(warning)
        
        return self.config
    
    def load_from_file(self, file_path: str) -> AppConfig:
        """
        Load configuration from a file.
        
        Args:
            file_path: Path to the configuration file
            
        Returns:
            Loaded configuration
        """
        # Validate file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        # Load configuration
        self.config = load_config(file_path)
        
        # Initialize logging
        initialize_logging(self.config)
        
        # Validate configuration
        self.warnings = validate_config(self.config)
        for warning in self.warnings:
            logger.warning(warning)
        
        return self.config
    
    def load_from_dict(self, config_dict: Dict[str, Any]) -> AppConfig:
        """
        Load configuration from a dictionary.
        
        Args:
            config_dict: Dictionary with configuration data
            
        Returns:
            Loaded configuration
        """
        # Load configuration
        self.config = load_config(args=config_dict)
        
        # Initialize logging
        initialize_logging(self.config)
        
        # Validate configuration
        self.warnings = validate_config(self.config)
        for warning in self.warnings:
            logger.warning(warning)
        
        return self.config
    
    def get_config(self) -> AppConfig:
        """
        Get the current configuration.
        
        Returns:
            Current configuration
            
        Raises:
            RuntimeError: If configuration has not been loaded
        """
        if self.config is None:
            raise RuntimeError("Configuration has not been loaded. Call load_* first.")
        
        return self.config
    
    def save_config(self, file_path: str, format: str = "json") -> None:
        """
        Save the current configuration to a file.
        
        Args:
            file_path: Path to save the configuration to
            format: Format to save the configuration in (json or yaml)
            
        Raises:
            RuntimeError: If configuration has not been loaded
        """
        if self.config is None:
            raise RuntimeError("Configuration has not been loaded. Call load_* first.")
        
        self.config.save(file_path)
        logger.info(f"Saved configuration to {file_path}")
    
    def validate(self) -> List[str]:
        """
        Validate the current configuration.
        
        Returns:
            List of validation warnings
            
        Raises:
            RuntimeError: If configuration has not been loaded
        """
        if self.config is None:
            raise RuntimeError("Configuration has not been loaded. Call load_* first.")
        
        self.warnings = validate_config(self.config)
        return self.warnings
    
    def verify_model(self) -> bool:
        """
        Verify that the configured model is available.
        
        Returns:
            True if the model is available, False otherwise
            
        Raises:
            RuntimeError: If configuration has not been loaded
        """
        if self.config is None:
            raise RuntimeError("Configuration has not been loaded. Call load_* first.")
        
        # Skip verification if disabled
        if not self.config.llm.verify_model:
            logger.info("Model verification disabled")
            return True
        
        # Verify model
        is_valid, error = validate_llm_model(self.config.llm.provider, self.config.llm.model_name)
        
        if not is_valid:
            logger.warning(f"Model verification failed: {error}")
            return False
        
        logger.info(f"Model {self.config.llm.provider}/{self.config.llm.model_name} verified successfully")
        return True
    
    def test_connections(self) -> Dict[str, bool]:
        """
        Test connections to configured APIs.
        
        Returns:
            Dictionary mapping API names to connection status
            
        Raises:
            RuntimeError: If configuration has not been loaded
        """
        if self.config is None:
            raise RuntimeError("Configuration has not been loaded. Call load_* first.")
        
        results = test_api_connections(self.config)
        
        # Log results
        for api, status in results.items():
            if status:
                logger.info(f"Connection to {api.upper()} API successful")
            else:
                logger.warning(f"Connection to {api.upper()} API failed")
        
        return results
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current configuration.
        
        Returns:
            Dictionary with configuration summary
            
        Raises:
            RuntimeError: If configuration has not been loaded
        """
        if self.config is None:
            raise RuntimeError("Configuration has not been loaded. Call load_* first.")
        
        # Build summary
        summary = {
            "provider": self.config.llm.provider,
            "model": self.config.llm.model_name,
            "operation_mode": self.config.operation_mode.current_mode,
            "budget": self.config.budget.budget,
            "optimize_costs": self.config.budget.optimize_costs,
            "original_paper": os.path.basename(self.config.files.original_paper_path),
            "reviewer_comments": [os.path.basename(path) for path in self.config.files.reviewer_comment_files],
            "output_dir": self.config.files.output_dir,
            "warnings": len(self.warnings),
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "version": self.config.version
        }
        
        return summary
    
    def print_summary(self) -> None:
        """
        Print a summary of the current configuration.
        
        Raises:
            RuntimeError: If configuration has not been loaded
        """
        if self.config is None:
            raise RuntimeError("Configuration has not been loaded. Call load_* first.")
        
        summary = self.get_summary()
        
        print("\nConfiguration Summary:")
        print("======================")
        print(f"Provider: {summary['provider'].upper()}")
        print(f"Model: {summary['model']}")
        print(f"Operation Mode: {summary['operation_mode']}")
        print(f"Budget: ${summary['budget']:.2f}")
        print(f"Optimize Costs: {'Yes' if summary['optimize_costs'] else 'No'}")
        print(f"Original Paper: {summary['original_paper']}")
        print(f"Reviewer Comments: {', '.join(summary['reviewer_comments'])}")
        print(f"Output Directory: {summary['output_dir']}")
        
        if summary['warnings'] > 0:
            print(f"\nWarnings: {summary['warnings']}")
            for warning in self.warnings[:5]:  # Show up to 5 warnings
                print(f"- {warning}")
            
            if len(self.warnings) > 5:
                print(f"... and {len(self.warnings) - 5} more warnings.")
        
        print(f"\nTimestamp: {summary['timestamp']}")
        print(f"Version: {summary['version']}")


# Global configuration manager instance
_manager: Optional[ConfigManager] = None


def get_manager() -> ConfigManager:
    """
    Get the global configuration manager instance.
    
    Returns:
        Configuration manager instance
    """
    global _manager
    
    if _manager is None:
        _manager = ConfigManager()
    
    return _manager