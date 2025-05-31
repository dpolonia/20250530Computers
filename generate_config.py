#!/usr/bin/env python3
"""
Generate configuration files for the paper revision tool.

This script generates sample configuration files and JSON schema for the
paper revision tool's configuration.
"""

import os
import argparse
import logging
from typing import Dict, Any, List

from src.config.schema import (
    generate_config_schema, save_schema_to_file,
    generate_sample_config, save_sample_config
)


def main():
    """Main function."""
    # Configure argument parser
    parser = argparse.ArgumentParser(
        description="Generate configuration files for the paper revision tool"
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory to save configuration files to"
    )
    parser.add_argument(
        "--schema",
        action="store_true",
        help="Generate JSON schema"
    )
    parser.add_argument(
        "--config",
        action="store_true",
        help="Generate sample configuration file"
    )
    parser.add_argument(
        "--format",
        choices=["json", "yaml"],
        default="json",
        help="Format for sample configuration file"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate all files"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate files
    if args.schema or args.all:
        schema_path = os.path.join(args.output_dir, "config-schema.json")
        save_schema_to_file(schema_path)
        print(f"Generated JSON schema: {schema_path}")
    
    if args.config or args.all:
        if args.format == "json":
            config_path = os.path.join(args.output_dir, "config-sample.json")
        else:
            config_path = os.path.join(args.output_dir, "config-sample.yaml")
        
        save_sample_config(config_path, args.format)
        print(f"Generated sample configuration: {config_path}")
    
    # If no action specified, show help
    if not (args.schema or args.config or args.all):
        parser.print_help()


if __name__ == "__main__":
    main()