#!/usr/bin/env python3
"""
Model Verification Tool

This tool checks if your LLM models are functioning correctly before using them
for paper revision. It verifies that the models can answer a simple factual question
correctly, which helps ensure they will provide reliable information.

Usage:
  python verify_models.py --provider anthropic|openai|google|all --model MODEL_NAME

Examples:
  python verify_models.py --provider anthropic  # Test default Anthropic model
  python verify_models.py --provider all        # Test default models for all providers
  python verify_models.py --provider openai --model gpt-4o  # Test specific model
"""

import os
import sys
import time
import argparse
import colorama
from colorama import Fore, Style
from typing import Dict, List, Any, Optional, Tuple
from dotenv import load_dotenv

# Initialize colorama for cross-platform colored terminal output
colorama.init()

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import utility modules
try:
    from src.utils.llm_client import get_llm_client
    from src.models.openai_models import get_openai_model_choices, get_max_tokens_for_model as get_openai_max_tokens
    from src.models.anthropic_models import get_claude_model_choices, get_max_tokens_for_model as get_claude_max_tokens
    from src.models.google_models import get_gemini_model_choices, get_max_tokens_for_model as get_gemini_max_tokens
except ImportError as e:
    print(f"{Fore.RED}Error importing required modules: {e}{Style.RESET_ALL}")
    print("Make sure you have all dependencies installed.")
    sys.exit(1)

def print_header(title: str) -> None:
    """Print a formatted header."""
    print(f"\n{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{title}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}")

def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n{Fore.YELLOW}{title}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{'-' * len(title)}{Style.RESET_ALL}")

def get_available_providers() -> List[str]:
    """Get a list of available providers based on API keys."""
    providers = []
    
    if os.getenv("ANTHROPIC_API_KEY"):
        providers.append("anthropic")
    
    if os.getenv("OPENAI_API_KEY"):
        providers.append("openai")
    
    if os.getenv("GOOGLE_API_KEY"):
        providers.append("google")
    
    return providers

def get_default_model(provider: str) -> str:
    """Get the default model for a provider."""
    if provider == "anthropic":
        return "claude-3-5-sonnet-20241022"
    elif provider == "openai":
        return "gpt-4o"
    elif provider == "google":
        return "gemini-1.5-pro"
    else:
        return "unknown"

def verify_model(provider: str, model_name: str) -> bool:
    """Verify a specific model.
    
    Args:
        provider: The provider name (anthropic, openai, google)
        model_name: The model name to verify
    
    Returns:
        True if model verification passed, False otherwise
    """
    print_section(f"Verifying {provider.upper()} model: {model_name}")
    
    try:
        # Check API key
        provider_env_vars = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "google": "GOOGLE_API_KEY"
        }
        
        env_var = provider_env_vars.get(provider, "API_KEY")
        
        if not os.getenv(env_var):
            print(f"{Fore.RED}Missing {env_var} in environment. Please add it to your .env file.{Style.RESET_ALL}")
            return False
        
        # Initialize client with verification
        print(f"Initializing {provider.upper()} client with model {model_name}...")
        client = get_llm_client(provider, model_name, verify=True)
        
        if not client.validate_api_key():
            print(f"{Fore.RED}Invalid API key for {provider}. The API key in your .env file is not working.{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Please check your {env_var} in .env file and make sure it is correct and not expired.{Style.RESET_ALL}")
            return False
        
        # Check verification result
        if client.verified:
            print(f"{Fore.GREEN}[✓] {provider.upper()} model {model_name} VERIFICATION PASSED{Style.RESET_ALL}")
            # Show usage stats
            stats = client.get_usage_statistics()
            print(f"\n{stats['summary']}")
            return True
        else:
            print(f"{Fore.RED}[✗] {provider.upper()} model {model_name} VERIFICATION FAILED{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}The model did not correctly answer when Christmas is (December 25th).{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}This may indicate the model is not functioning correctly or may provide inaccurate information.{Style.RESET_ALL}")
            return False
            
    except Exception as e:
        print(f"{Fore.RED}Error verifying {provider} model {model_name}: {str(e)}{Style.RESET_ALL}")
        return False

def main() -> None:
    """Run the model verification tool."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Model Verification Tool")
    parser.add_argument("--provider", choices=["anthropic", "openai", "google", "all"], default="all",
                      help="Which provider to verify (default: all available)")
    parser.add_argument("--model", help="Specific model to verify (optional)")
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Print header
    print_header("MODEL VERIFICATION TOOL")
    
    # Determine which providers to test
    available_providers = get_available_providers()
    
    if not available_providers:
        print(f"{Fore.RED}Error: No API keys found in environment.{Style.RESET_ALL}")
        print("Please set at least one of the following in your .env file:")
        print("  - ANTHROPIC_API_KEY")
        print("  - OPENAI_API_KEY")
        print("  - GOOGLE_API_KEY")
        sys.exit(1)
    
    if args.provider == "all":
        providers_to_test = available_providers
    elif args.provider in available_providers:
        providers_to_test = [args.provider]
    else:
        print(f"{Fore.RED}Error: API key for {args.provider} not found.{Style.RESET_ALL}")
        sys.exit(1)
    
    # Verify models for each provider
    results = {}
    for provider in providers_to_test:
        model = args.model or get_default_model(provider)
        results[f"{provider}/{model}"] = verify_model(provider, model)
    
    # Print summary
    print_header("VERIFICATION SUMMARY")
    
    passed = []
    failed = []
    
    for model_key, result in results.items():
        if result:
            passed.append(model_key)
        else:
            failed.append(model_key)
    
    if passed:
        print(f"{Fore.GREEN}Models that passed verification:{Style.RESET_ALL}")
        for model in passed:
            print(f"  - {model}")
    
    if failed:
        print(f"\n{Fore.RED}Models that failed verification:{Style.RESET_ALL}")
        for model in failed:
            print(f"  - {model}")
        
        print(f"\n{Fore.YELLOW}Recommendation: Run the paper revision tool with one of the verified models.{Style.RESET_ALL}")
    
    # Exit with appropriate status code
    if failed:
        print(f"\n{Fore.YELLOW}Warning: Some models failed verification. You can still use them but proceed with caution.{Style.RESET_ALL}")
        sys.exit(0)  # Exit with success to allow continuing
    else:
        print(f"\n{Fore.GREEN}All models passed verification. You can proceed with confidence.{Style.RESET_ALL}")
        sys.exit(0)

if __name__ == "__main__":
    main()