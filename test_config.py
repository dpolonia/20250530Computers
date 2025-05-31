#!/usr/bin/env python3
"""
Test configuration system for the paper revision tool.

This script tests the configuration system by loading and validating
configuration from different sources.
"""

import os
import sys
import argparse
import logging
from pprint import pprint

from src.config import (
    AppConfig, load_env_files, load_config, initialize_logging,
    validate_config, validate_file_exists, validate_api_key,
    validate_llm_model, validate_budget_settings, validate_api_config,
    verify_model_compatibility, test_api_connections
)
from src.config.manager import get_manager


def test_load_config_from_file(file_path: str) -> AppConfig:
    """
    Test loading configuration from a file.
    
    Args:
        file_path: Path to the configuration file
        
    Returns:
        Loaded configuration
    """
    print(f"\nLoading configuration from file: {file_path}")
    
    # Initialize configuration manager
    manager = get_manager()
    
    # Load configuration
    config = manager.load_from_file(file_path)
    
    print("Configuration loaded successfully")
    
    return config


def test_load_config_from_env() -> AppConfig:
    """
    Test loading configuration from environment variables.
    
    Returns:
        Loaded configuration
    """
    print("\nLoading configuration from environment variables")
    
    # Load environment variables
    load_env_files()
    
    # Load configuration
    config = load_config()
    
    print("Configuration loaded successfully")
    
    return config


def test_validate_config(config: AppConfig) -> None:
    """
    Test validating configuration.
    
    Args:
        config: Configuration to validate
    """
    print("\nValidating configuration")
    
    # Validate configuration
    warnings = validate_config(config)
    
    if warnings:
        print(f"Found {len(warnings)} validation warnings:")
        for warning in warnings:
            print(f"- {warning}")
    else:
        print("Configuration is valid")


def test_verify_model(config: AppConfig) -> None:
    """
    Test verifying model compatibility.
    
    Args:
        config: Configuration with model settings
    """
    print("\nVerifying model compatibility")
    
    # Verify model compatibility
    is_compatible, warnings = verify_model_compatibility(config)
    
    if is_compatible:
        print(f"Model {config.llm.provider}/{config.llm.model_name} is compatible")
    else:
        print(f"Model {config.llm.provider}/{config.llm.model_name} is not compatible")
    
    if warnings:
        print(f"Found {len(warnings)} compatibility warnings:")
        for warning in warnings:
            print(f"- {warning}")


def test_validate_llm_model(provider: str, model_name: str) -> None:
    """
    Test validating LLM model availability.
    
    Args:
        provider: Provider name
        model_name: Model name
    """
    print(f"\nValidating model {provider}/{model_name}")
    
    # Validate model
    is_valid, error = validate_llm_model(provider, model_name)
    
    if is_valid:
        print(f"Model {provider}/{model_name} is valid")
    else:
        print(f"Model {provider}/{model_name} is not valid: {error}")


def test_api_connections(config: AppConfig) -> None:
    """
    Test API connections.
    
    Args:
        config: Configuration with API settings
    """
    print("\nTesting API connections")
    
    # Test API connections
    results = test_api_connections(config)
    
    for api, status in results.items():
        print(f"- {api.upper()}: {'Connected' if status else 'Failed'}")


def test_manager_summary(config: AppConfig) -> None:
    """
    Test configuration manager summary.
    
    Args:
        config: Configuration to summarize
    """
    print("\nConfiguration summary")
    
    # Get manager
    manager = get_manager()
    
    # Print summary
    manager.print_summary()


def test_save_config(config: AppConfig, output_path: str) -> None:
    """
    Test saving configuration to a file.
    
    Args:
        config: Configuration to save
        output_path: Path to save the configuration to
    """
    print(f"\nSaving configuration to {output_path}")
    
    # Save configuration
    config.save(output_path)
    
    print(f"Configuration saved to {output_path}")


def main():
    """Main function."""
    # Configure argument parser
    parser = argparse.ArgumentParser(
        description="Test configuration system for the paper revision tool"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the configuration file"
    )
    parser.add_argument(
        "--load",
        action="store_true",
        help="Test loading configuration"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Test validating configuration"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Test verifying model compatibility"
    )
    parser.add_argument(
        "--test-api",
        action="store_true",
        help="Test API connections"
    )
    parser.add_argument(
        "--save",
        help="Path to save the configuration to"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all tests"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Load configuration
    try:
        if args.load or args.all:
            config = test_load_config_from_file(args.config)
        else:
            # Initialize configuration manager
            manager = get_manager()
            
            # Load configuration from file
            config = manager.load_from_file(args.config)
        
        # Run tests
        if args.validate or args.all:
            test_validate_config(config)
        
        if args.verify or args.all:
            test_verify_model(config)
        
        if args.test_api or args.all:
            test_api_connections(config)
        
        # Print configuration summary
        test_manager_summary(config)
        
        # Save configuration if requested
        if args.save:
            test_save_config(config, args.save)
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())