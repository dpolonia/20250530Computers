"""
Command-line interface configuration.

This module provides functions for configuring and parsing command-line
arguments for the paper revision tool.
"""

import os
import sys
import argparse
import logging
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass

from src.config.configuration import AppConfig
from src.config.validation import validate_config
from src.core.constants import OPERATION_MODES

# Configure logging for the CLI module
logger = logging.getLogger(__name__)


@dataclass
class CommandDefinition:
    """Definition of a command-line argument."""
    
    name: str
    flags: List[str]
    help: str
    type: Optional[type] = None
    default: Any = None
    required: bool = False
    choices: Optional[List[str]] = None
    nargs: Optional[str] = None
    action: Optional[str] = None
    dest: Optional[str] = None
    metavar: Optional[str] = None
    
    def add_to_parser(self, parser: argparse.ArgumentParser) -> None:
        """
        Add this command to an argument parser.
        
        Args:
            parser: ArgumentParser to add the command to
        """
        kwargs = {
            "help": self.help,
            "required": self.required,
        }
        
        if self.type is not None:
            kwargs["type"] = self.type
        
        if self.default is not None:
            kwargs["default"] = self.default
        
        if self.choices is not None:
            kwargs["choices"] = self.choices
        
        if self.nargs is not None:
            kwargs["nargs"] = self.nargs
        
        if self.action is not None:
            kwargs["action"] = self.action
        
        if self.dest is not None:
            kwargs["dest"] = self.dest
        
        if self.metavar is not None:
            kwargs["metavar"] = self.metavar
        
        parser.add_argument(*self.flags, **kwargs)


# Define command-line arguments
COMMANDS = [
    CommandDefinition(
        name="paper",
        flags=["--paper"],
        help="Path to the original paper PDF",
        type=str,
        required=True,
    ),
    CommandDefinition(
        name="reviewer_comments",
        flags=["--reviewer_comments"],
        help="Paths to reviewer comment files",
        nargs="+",
        required=True,
    ),
    CommandDefinition(
        name="editor_letter",
        flags=["--editor_letter"],
        help="Path to editor letter file",
        type=str,
    ),
    CommandDefinition(
        name="output_dir",
        flags=["--output_dir"],
        help="Directory for output files",
        type=str,
    ),
    CommandDefinition(
        name="provider",
        flags=["--provider"],
        help="LLM provider (default: anthropic)",
        type=str,
        default="anthropic",
        choices=["anthropic", "openai", "google"],
    ),
    CommandDefinition(
        name="model",
        flags=["--model"],
        help="LLM model name (default: claude-3-opus-20240229)",
        type=str,
        default="claude-3-opus-20240229",
    ),
    CommandDefinition(
        name="mode",
        flags=["--mode"],
        help="Operation mode (default: finetuning)",
        type=str,
        default="finetuning",
        choices=list(OPERATION_MODES.keys()),
    ),
    CommandDefinition(
        name="budget",
        flags=["--budget"],
        help="Maximum budget in dollars (default depends on operation mode)",
        type=float,
    ),
    CommandDefinition(
        name="optimize_costs",
        flags=["--optimize_costs"],
        help="Optimize for lower costs",
        action="store_true",
    ),
    CommandDefinition(
        name="verbose",
        flags=["--verbose", "-v"],
        help="Enable verbose logging",
        action="store_true",
    ),
    CommandDefinition(
        name="full",
        flags=["--full"],
        help="Run the full process automatically",
        action="store_true",
    ),
    CommandDefinition(
        name="interactive",
        flags=["--interactive", "-i"],
        help="Run in interactive mode",
        action="store_true",
    ),
    CommandDefinition(
        name="api",
        flags=["--api"],
        help="Enable external API integration (comma-separated list: scopus,wos)",
        type=str,
    ),
    CommandDefinition(
        name="key",
        flags=["--key"],
        help="API key for external APIs",
        type=str,
    ),
    CommandDefinition(
        name="config",
        flags=["--config"],
        help="Path to configuration file",
        type=str,
    ),
    CommandDefinition(
        name="no_verify",
        flags=["--no_verify"],
        help="Skip model verification",
        action="store_true",
    ),
    CommandDefinition(
        name="run_id",
        flags=["--run_id"],
        help="Run ID for tracking",
        type=str,
    ),
    CommandDefinition(
        name="version",
        flags=["--version"],
        help="Show version information and exit",
        action="version",
        version="%(prog)s 1.0.0",
    ),
]


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create an argument parser for the paper revision tool.
    
    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description="Paper Revision Tool - Automate paper revisions with AI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Add commands to parser
    for command in COMMANDS:
        command.add_to_parser(parser)
    
    return parser


def parse_args(args: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Parse command-line arguments.
    
    Args:
        args: Optional list of command-line arguments. If None, uses sys.argv.
        
    Returns:
        Dictionary of parsed arguments
    """
    parser = create_argument_parser()
    parsed_args = parser.parse_args(args)
    
    # Convert to dictionary
    return vars(parsed_args)


def process_args(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process and validate command-line arguments.
    
    Args:
        args: Dictionary of parsed arguments
        
    Returns:
        Processed arguments dictionary
    """
    # Process API argument (convert comma-separated string to list)
    if "api" in args and args["api"]:
        if isinstance(args["api"], str) and "," in args["api"]:
            args["api"] = [api.strip() for api in args["api"].split(",")]
    
    # Normalize file paths
    for path_arg in ["paper", "editor_letter", "output_dir", "config"]:
        if path_arg in args and args[path_arg]:
            args[path_arg] = os.path.abspath(args[path_arg])
    
    if "reviewer_comments" in args and args["reviewer_comments"]:
        args["reviewer_comments"] = [os.path.abspath(path) for path in args["reviewer_comments"]]
    
    # If mode is specified, apply mode-specific settings
    if "mode" in args and args["mode"]:
        mode = args["mode"]
        if mode in OPERATION_MODES:
            mode_settings = OPERATION_MODES[mode]
            
            # Apply mode-specific settings if not explicitly set in args
            if "optimize_costs" not in args or args["optimize_costs"] is None:
                args["optimize_costs"] = mode_settings.get("optimize_costs", False)
            
            if "budget" not in args or args["budget"] is None:
                args["budget"] = mode_settings.get("budget", 10.0)
            
            # Set recommended model if provider is specified but model is not
            if "provider" in args and args["provider"] and ("model" not in args or not args["model"]):
                provider = args["provider"]
                provider_recommendations = mode_settings.get("provider_recommendations", {})
                if provider in provider_recommendations:
                    args["model"] = provider_recommendations[provider]
    
    return args


def apply_args_to_config(args: Dict[str, Any], config: AppConfig) -> AppConfig:
    """
    Apply command-line arguments to configuration.
    
    Args:
        args: Dictionary of processed arguments
        config: Configuration to update
        
    Returns:
        Updated configuration
    """
    # LLM configuration
    if "provider" in args and args["provider"]:
        config.llm.provider = args["provider"]
    
    if "model" in args and args["model"]:
        config.llm.model_name = args["model"]
    
    if "no_verify" in args:
        config.llm.verify_model = not args["no_verify"]
    
    # Budget configuration
    if "budget" in args and args["budget"] is not None:
        config.budget.budget = args["budget"]
    
    if "optimize_costs" in args:
        config.budget.optimize_costs = args["optimize_costs"]
    
    # File configuration
    if "paper" in args and args["paper"]:
        config.files.original_paper_path = args["paper"]
    
    if "reviewer_comments" in args and args["reviewer_comments"]:
        config.files.reviewer_comment_files = args["reviewer_comments"]
    
    if "editor_letter" in args and args["editor_letter"]:
        config.files.editor_letter_path = args["editor_letter"]
    
    if "output_dir" in args and args["output_dir"]:
        config.files.output_dir = args["output_dir"]
    
    # Logging configuration
    if "verbose" in args:
        config.logging.level = "DEBUG" if args["verbose"] else "INFO"
    
    # API configuration
    if "api" in args and args["api"]:
        if isinstance(args["api"], str):
            config.api.enabled_apis = [args["api"]]
        else:
            config.api.enabled_apis = args["api"]
    
    if "key" in args and args["key"]:
        # Assign key based on enabled APIs
        if "scopus" in config.api.enabled_apis:
            config.api.scopus_api_key = args["key"]
    
    # Operation mode configuration
    if "mode" in args and args["mode"]:
        config.operation_mode.current_mode = args["mode"]
    
    # Application-wide settings
    if "interactive" in args:
        config.interactive = args["interactive"]
    
    if "full" in args:
        config.interactive = not args["full"]
    
    if "run_id" in args and args["run_id"]:
        config.run_id = args["run_id"]
    
    # Apply settings from the current operation mode
    config.apply_operation_mode()
    
    return config


def validate_args(args: Dict[str, Any]) -> List[str]:
    """
    Validate command-line arguments and return a list of errors.
    
    Args:
        args: Dictionary of processed arguments
        
    Returns:
        List of error messages
    """
    errors = []
    
    # Validate paper path
    if "paper" in args and args["paper"]:
        if not os.path.exists(args["paper"]):
            errors.append(f"Paper file not found: {args['paper']}")
    else:
        errors.append("Paper path is required")
    
    # Validate reviewer comment paths
    if "reviewer_comments" in args and args["reviewer_comments"]:
        for i, path in enumerate(args["reviewer_comments"]):
            if not os.path.exists(path):
                errors.append(f"Reviewer comment file {i+1} not found: {path}")
    else:
        errors.append("At least one reviewer comment file is required")
    
    # Validate editor letter path
    if "editor_letter" in args and args["editor_letter"]:
        if not os.path.exists(args["editor_letter"]):
            errors.append(f"Editor letter file not found: {args['editor_letter']}")
    
    # Validate output directory
    if "output_dir" in args and args["output_dir"]:
        try:
            os.makedirs(args["output_dir"], exist_ok=True)
        except Exception as e:
            errors.append(f"Could not create output directory: {e}")
    
    # Validate operation mode
    if "mode" in args and args["mode"]:
        if args["mode"] not in OPERATION_MODES:
            errors.append(f"Invalid operation mode: {args['mode']}. Valid modes: {', '.join(OPERATION_MODES.keys())}")
    
    # Validate budget
    if "budget" in args and args["budget"] is not None:
        if args["budget"] <= 0:
            errors.append(f"Budget must be greater than 0. Got: {args['budget']}")
    
    # Validate provider
    if "provider" in args and args["provider"]:
        valid_providers = {"anthropic", "openai", "google"}
        if args["provider"] not in valid_providers:
            errors.append(f"Invalid provider: {args['provider']}. Valid providers: {', '.join(valid_providers)}")
    
    # Validate API
    if "api" in args and args["api"]:
        valid_apis = {"scopus", "wos"}
        apis = args["api"]
        if isinstance(apis, str):
            apis = [api.strip() for api in apis.split(",")]
        
        for api in apis:
            if api not in valid_apis:
                errors.append(f"Invalid API: {api}. Valid APIs: {', '.join(valid_apis)}")
    
    # Validate config file
    if "config" in args and args["config"]:
        if not os.path.exists(args["config"]):
            errors.append(f"Configuration file not found: {args['config']}")
    
    return errors


def run_with_args(
    args: Optional[List[str]] = None,
    runner_func: Optional[callable] = None
) -> int:
    """
    Run the paper revision tool with command-line arguments.
    
    Args:
        args: Optional list of command-line arguments. If None, uses sys.argv.
        runner_func: Function to run with the configuration (takes AppConfig argument)
        
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    # Parse and process arguments
    parsed_args = parse_args(args)
    processed_args = process_args(parsed_args)
    
    # Check for errors in arguments
    errors = validate_args(processed_args)
    if errors:
        for error in errors:
            print(f"Error: {error}")
        return 1
    
    # Load configuration
    from src.config.configuration import load_config
    config_file = processed_args.get("config")
    config = load_config(config_file, processed_args)
    
    # Initialize logging
    from src.config.configuration import initialize_logging
    initialize_logging(config)
    
    # Validate configuration
    warnings = validate_config(config)
    for warning in warnings:
        logger.warning(warning)
    
    # Run the function with the configuration
    if runner_func:
        try:
            return runner_func(config)
        except Exception as e:
            logger.error(f"Error running application: {e}", exc_info=True)
            return 1
    
    return 0