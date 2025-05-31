#!/usr/bin/env python3
"""
Comprehensive demonstration of the model verification system.

This script provides a complete demonstration of the model verification system,
showing how it works with all supported models and different verification scenarios.
"""

import os
import sys
import time
import argparse
import datetime
import colorama
from colorama import Fore, Style
from src.utils.llm_client import get_llm_client
from dotenv import load_dotenv

# Initialize colorama
colorama.init()

def print_header(title):
    """Print a formatted header."""
    print(f"\n{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{title}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}")

def print_section(title):
    """Print a section header."""
    print(f"\n{Fore.YELLOW}{title}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{'-' * len(title)}{Style.RESET_ALL}")

def get_available_providers():
    """Get a list of available providers based on API keys."""
    providers = []
    
    if os.getenv("ANTHROPIC_API_KEY"):
        providers.append("anthropic")
    
    if os.getenv("OPENAI_API_KEY"):
        providers.append("openai")
    
    if os.getenv("GOOGLE_API_KEY"):
        providers.append("google")
    
    return providers

def get_default_model(provider):
    """Get the default model for a provider."""
    default_models = {
        "anthropic": "claude-3-sonnet-20240229",
        "openai": "gpt-4-turbo",
        "google": "gemini-1.5-pro"
    }
    return default_models.get(provider, "unknown")

def test_model(provider, model_name, verify=True):
    """Test a specific model with optional verification."""
    print_section(f"Testing {provider.upper()} model: {model_name}")
    
    try:
        # Initialize client with or without verification
        client = get_llm_client(provider, model_name, verify=verify)
        
        if verify:
            if client.verified:
                print(f"{Fore.GREEN}[✓] Model verification passed!{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}[✗] Model verification failed!{Style.RESET_ALL}")
                choice = input("Continue anyway? (y/n): ")
                if choice.lower() != 'y':
                    print("Aborting test.")
                    return
        else:
            print(f"{Fore.YELLOW}[!] Model verification skipped{Style.RESET_ALL}")
        
        # Ask the model a simple question
        print("\nSending a test question to the model...")
        response = client.get_completion(
            prompt="What are the primary colors?",
            system_prompt="Answer concisely in a single sentence.",
            max_tokens=100
        )
        
        print(f"\n{Fore.GREEN}Response:{Style.RESET_ALL} {response}")
        
        # Show usage statistics
        print_section("Usage Statistics")
        stats = client.get_usage_statistics()
        print(stats["summary"])
        
        return client
        
    except Exception as e:
        print(f"{Fore.RED}Error testing {provider} model {model_name}: {str(e)}{Style.RESET_ALL}")
        return None

def simulate_verification_failure(client):
    """Simulate a verification failure with a model."""
    if not client:
        return
    
    print_section(f"Simulating verification failure for {client.provider.upper()} {client.model}")
    
    # Store original verification method
    original_verify = client.verify_model_accuracy
    
    # Mock verification to always fail
    def mock_verification(*args, **kwargs):
        print(f"\n[!] {client.provider.upper()} {client.model} VERIFICATION FAILED (SIMULATED)")
        print("Question: In what day is Christmas?")
        print("Response: Christmas is a holiday (simulated incorrect response)")
        print("ALERT: Model did not answer 'December 25th' or similar")
        client.verified = False
        return False
    
    try:
        # Replace with mock implementation
        client.verify_model_accuracy = mock_verification
        
        # Force a new verification that will now fail
        client.verified = False
        
        print("\nAttempting to use model after simulated verification failure...")
        print("You will be asked if you want to continue despite verification failure.")
        print("Enter 'y' to continue or 'n' to abort.")
        
        try:
            response = client.get_completion(
                prompt="What is the capital of Italy?",
                system_prompt="Answer in one word only.",
                max_tokens=10
            )
            print(f"\n{Fore.GREEN}Response:{Style.RESET_ALL} {response}")
            print(f"{Fore.YELLOW}(Continued despite verification failure){Style.RESET_ALL}")
        except RuntimeError as e:
            print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
            print("Verification failure was handled as expected (abort on failure).")
            
    finally:
        # Restore original verification method
        client.verify_model_accuracy = original_verify
        client.verified = True

def main():
    """Run the model verification demonstration."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Model Verification Demo")
    parser.add_argument("--provider", choices=["anthropic", "openai", "google", "all"], default="all",
                      help="Which provider to test (default: all available)")
    parser.add_argument("--model", help="Specific model to test (optional)")
    parser.add_argument("--skip-verify", action="store_true", help="Skip initial verification")
    parser.add_argument("--simulate-failure", action="store_true", help="Simulate verification failure")
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Print header
    now = datetime.datetime.now()
    print_header(f"MODEL VERIFICATION DEMONSTRATION - {now.strftime('%Y-%m-%d %H:%M:%S')}")
    
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
    
    # Run tests for each provider
    clients = []
    for provider in providers_to_test:
        model = args.model or get_default_model(provider)
        client = test_model(provider, model, verify=not args.skip_verify)
        if client:
            clients.append(client)
    
    # Simulate verification failure if requested
    if args.simulate_failure and clients:
        # Use the first client for simulation
        simulate_verification_failure(clients[0])
    
    print(f"\n{Fore.GREEN}Demonstration completed successfully.{Style.RESET_ALL}")

if __name__ == "__main__":
    main()