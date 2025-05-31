#!/usr/bin/env python3
"""
Credential management utility for Paper Revision System.

This script provides a command-line interface for securely managing API keys
and other credentials used by the system.
"""

import os
import sys
import argparse
import getpass
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.security.credential_manager import (
    get_credential_manager, 
    CredentialManager
)
from src.security.input_validator import (
    validate_api_key,
    ValidationError
)


def setup_argparse():
    """Set up command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Securely manage API keys and other credentials",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Store a new Anthropic API key
  python manage_credentials.py store anthropic
  
  # Verify existing API keys
  python manage_credentials.py verify
  
  # Export all API keys to environment variables
  python manage_credentials.py export
  
  # Import API keys from environment variables
  python manage_credentials.py import
  
  # View masked API keys
  python manage_credentials.py list
  
  # Delete a stored API key
  python manage_credentials.py delete openai
"""
    )
    
    # Add subparsers for commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # 'store' command
    store_parser = subparsers.add_parser("store", help="Store a new API key")
    store_parser.add_argument(
        "provider",
        choices=["anthropic", "openai", "google", "all"],
        help="Provider for the API key"
    )
    store_parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Overwrite existing API key if it exists"
    )
    
    # 'verify' command
    verify_parser = subparsers.add_parser("verify", help="Verify API keys")
    verify_parser.add_argument(
        "provider",
        nargs="?",
        choices=["anthropic", "openai", "google", "all"],
        default="all",
        help="Provider to verify (default: all)"
    )
    
    # 'export' command
    export_parser = subparsers.add_parser(
        "export", 
        help="Export API keys to environment variables"
    )
    export_parser.add_argument(
        "provider",
        nargs="?",
        choices=["anthropic", "openai", "google", "all"],
        default="all",
        help="Provider to export (default: all)"
    )
    
    # 'import' command
    import_parser = subparsers.add_parser(
        "import", 
        help="Import API keys from environment variables"
    )
    import_parser.add_argument(
        "provider",
        nargs="?",
        choices=["anthropic", "openai", "google", "all"],
        default="all",
        help="Provider to import (default: all)"
    )
    import_parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Overwrite existing API keys if they exist"
    )
    
    # 'list' command
    list_parser = subparsers.add_parser("list", help="List stored API keys")
    
    # 'delete' command
    delete_parser = subparsers.add_parser("delete", help="Delete a stored API key")
    delete_parser.add_argument(
        "provider",
        choices=["anthropic", "openai", "google", "all"],
        help="Provider for the API key to delete"
    )
    delete_parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Delete without confirmation"
    )
    
    return parser


def get_provider_key_type(provider):
    """Get the credential manager key type for a provider."""
    if provider == "anthropic":
        return CredentialManager.ANTHROPIC_API_KEY
    elif provider == "openai":
        return CredentialManager.OPENAI_API_KEY
    elif provider == "google":
        return CredentialManager.GOOGLE_API_KEY
    else:
        return None


def get_provider_env_var(provider):
    """Get the environment variable name for a provider."""
    if provider == "anthropic":
        return "ANTHROPIC_API_KEY"
    elif provider == "openai":
        return "OPENAI_API_KEY"
    elif provider == "google":
        return "GOOGLE_API_KEY"
    else:
        return None


def store_api_key(credential_manager, provider, force=False):
    """Store an API key for a provider."""
    print(f"\nStoring {provider.upper()} API key")
    key_type = get_provider_key_type(provider)
    
    # Check if key already exists
    existing_key = credential_manager.get_credential(key_type)
    if existing_key and not force:
        overwrite = input(f"API key for {provider.upper()} already exists. Overwrite? (y/n): ")
        if overwrite.lower() != "y":
            print(f"Skipping {provider.upper()} API key")
            return False
    
    # Get API key from user input (hiding input)
    api_key = getpass.getpass(f"Enter {provider.upper()} API key: ")
    
    # Validate API key
    try:
        api_key = validate_api_key(api_key, min_length=8)
    except ValidationError as e:
        print(f"Invalid API key: {e}")
        return False
    
    # Store the API key
    result = credential_manager.store_credential(key_type, api_key, overwrite=True)
    
    if result:
        print(f"{provider.upper()} API key stored successfully")
    else:
        print(f"Failed to store {provider.upper()} API key")
        
    return result


def verify_api_key(credential_manager, provider):
    """Verify an API key for a provider."""
    print(f"\nVerifying {provider.upper()} API key")
    key_type = get_provider_key_type(provider)
    
    # Check if key exists
    api_key = credential_manager.get_credential(key_type)
    if not api_key:
        print(f"No API key found for {provider.upper()}")
        return False
    
    # Import the appropriate model module
    if provider == "anthropic":
        from src.models.anthropic_models import validate_api_key as validate_func
    elif provider == "openai":
        from src.models.openai_models import validate_api_key as validate_func
    elif provider == "google":
        from src.models.google_models import validate_api_key as validate_func
    
    # Validate the API key
    valid = credential_manager.validate_credential(key_type, validate_func)
    
    if valid:
        print(f"{provider.upper()} API key is valid")
    else:
        print(f"{provider.upper()} API key is invalid")
        
    return valid


def export_api_key(credential_manager, provider):
    """Export an API key to an environment variable."""
    print(f"\nExporting {provider.upper()} API key to environment variable")
    key_type = get_provider_key_type(provider)
    env_var = get_provider_env_var(provider)
    
    # Check if key exists
    api_key = credential_manager.get_credential(key_type)
    if not api_key:
        print(f"No API key found for {provider.upper()}")
        return False
    
    # Export to environment variable
    os.environ[env_var] = api_key
    print(f"Exported {provider.upper()} API key to {env_var}")
    
    return True


def import_api_key(credential_manager, provider, force=False):
    """Import an API key from an environment variable."""
    print(f"\nImporting {provider.upper()} API key from environment variable")
    key_type = get_provider_key_type(provider)
    env_var = get_provider_env_var(provider)
    
    # Check if environment variable exists
    api_key = os.getenv(env_var)
    if not api_key:
        print(f"No API key found in environment variable {env_var}")
        return False
    
    # Check if key already exists
    existing_key = credential_manager.get_credential(key_type)
    if existing_key and not force:
        print(f"API key for {provider.upper()} already exists. Use --force to overwrite.")
        return False
    
    # Import the API key
    result = credential_manager.store_credential(key_type, api_key, overwrite=force)
    
    if result:
        print(f"{provider.upper()} API key imported successfully")
    else:
        print(f"Failed to import {provider.upper()} API key")
        
    return result


def list_api_keys(credential_manager):
    """List all stored API keys."""
    print("\nStored API keys:")
    
    # Check Anthropic API key
    anthropic_key = credential_manager.get_credential(CredentialManager.ANTHROPIC_API_KEY)
    if anthropic_key:
        # Mask the key (show first 4 and last 4 characters)
        masked_key = anthropic_key[:4] + "****" + anthropic_key[-4:] if len(anthropic_key) > 8 else "****"
        print(f"ANTHROPIC_API_KEY: {masked_key}")
    else:
        print("ANTHROPIC_API_KEY: Not set")
    
    # Check OpenAI API key
    openai_key = credential_manager.get_credential(CredentialManager.OPENAI_API_KEY)
    if openai_key:
        masked_key = openai_key[:4] + "****" + openai_key[-4:] if len(openai_key) > 8 else "****"
        print(f"OPENAI_API_KEY: {masked_key}")
    else:
        print("OPENAI_API_KEY: Not set")
    
    # Check Google API key
    google_key = credential_manager.get_credential(CredentialManager.GOOGLE_API_KEY)
    if google_key:
        masked_key = google_key[:4] + "****" + google_key[-4:] if len(google_key) > 8 else "****"
        print(f"GOOGLE_API_KEY: {masked_key}")
    else:
        print("GOOGLE_API_KEY: Not set")
    
    return True


def delete_api_key(credential_manager, provider, force=False):
    """Delete a stored API key."""
    print(f"\nDeleting {provider.upper()} API key")
    key_type = get_provider_key_type(provider)
    
    # Check if key exists
    api_key = credential_manager.get_credential(key_type)
    if not api_key:
        print(f"No API key found for {provider.upper()}")
        return False
    
    # Confirm deletion
    if not force:
        confirm = input(f"Are you sure you want to delete the {provider.upper()} API key? (y/n): ")
        if confirm.lower() != "y":
            print(f"Aborted deletion of {provider.upper()} API key")
            return False
    
    # Delete the API key
    result = credential_manager.clear_credential(key_type)
    
    if result:
        print(f"{provider.upper()} API key deleted successfully")
    else:
        print(f"Failed to delete {provider.upper()} API key")
        
    return result


def main():
    """Main function for the credential management utility."""
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Get credential manager
    credential_manager = get_credential_manager()
    
    # Process command
    if args.command == "store":
        if args.provider == "all":
            providers = ["anthropic", "openai", "google"]
            results = []
            for provider in providers:
                results.append(store_api_key(credential_manager, provider, args.force))
            return all(results)
        else:
            return store_api_key(credential_manager, args.provider, args.force)
    
    elif args.command == "verify":
        if args.provider == "all":
            providers = ["anthropic", "openai", "google"]
            results = []
            for provider in providers:
                results.append(verify_api_key(credential_manager, provider))
            return any(results)  # Success if at least one key is valid
        else:
            return verify_api_key(credential_manager, args.provider)
    
    elif args.command == "export":
        if args.provider == "all":
            providers = ["anthropic", "openai", "google"]
            results = []
            for provider in providers:
                results.append(export_api_key(credential_manager, provider))
            return any(results)  # Success if at least one key is exported
        else:
            return export_api_key(credential_manager, args.provider)
    
    elif args.command == "import":
        if args.provider == "all":
            providers = ["anthropic", "openai", "google"]
            results = []
            for provider in providers:
                results.append(import_api_key(credential_manager, provider, args.force))
            return any(results)  # Success if at least one key is imported
        else:
            return import_api_key(credential_manager, args.provider, args.force)
    
    elif args.command == "list":
        return list_api_keys(credential_manager)
    
    elif args.command == "delete":
        if args.provider == "all":
            providers = ["anthropic", "openai", "google"]
            results = []
            for provider in providers:
                results.append(delete_api_key(credential_manager, provider, args.force))
            return all(results)
        else:
            return delete_api_key(credential_manager, args.provider, args.force)
    
    else:
        parser.print_help()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)