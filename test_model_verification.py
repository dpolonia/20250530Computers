#!/usr/bin/env python3
"""
Test script for the model verification system.

This script demonstrates the enhanced model verification system that checks
model accuracy before every API call. It demonstrates:

1. The verification test on initialization (performed once)
2. A subsequent call that should skip verification because model was already verified
3. A forced verification (by setting verified=False)
4. A verification failure scenario (simulated)
5. How to continue despite verification failure
"""

import os
import sys
import time
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

def main():
    """Run the model verification tests."""
    # Load environment variables from .env file
    load_dotenv()
    
    # Make sure we have at least one API key
    has_api_key = False
    providers_to_test = []
    
    if os.getenv("ANTHROPIC_API_KEY"):
        has_api_key = True
        providers_to_test.append(("anthropic", "claude-3-sonnet-20240229"))
    
    if os.getenv("OPENAI_API_KEY"):
        has_api_key = True
        providers_to_test.append(("openai", "gpt-4-turbo"))
    
    if os.getenv("GOOGLE_API_KEY"):
        has_api_key = True
        providers_to_test.append(("google", "gemini-1.5-pro"))
    
    if not has_api_key:
        print(f"{Fore.RED}Error: No API keys found in environment.{Style.RESET_ALL}")
        print("Please set at least one of the following in your .env file:")
        print("  - ANTHROPIC_API_KEY")
        print("  - OPENAI_API_KEY")
        print("  - GOOGLE_API_KEY")
        sys.exit(1)
    
    # Run tests for each available provider
    for provider, model in providers_to_test:
        run_test_for_provider(provider, model)

def run_test_for_provider(provider, model):
    """Run the verification tests for a specific provider."""
    print_header(f"TESTING MODEL VERIFICATION FOR {provider.upper()} {model}")
    
    try:
        # Test 1: Initialize with verification
        print(f"\n{Fore.YELLOW}Test 1: Initialize with verification{Style.RESET_ALL}")
        print("This will test if the model correctly answers when Christmas is...")
        client = get_llm_client(provider, model, verify=True)
        
        # Test 2: Use the model (should skip verification)
        print(f"\n{Fore.YELLOW}Test 2: Using model after verification{Style.RESET_ALL}")
        print("This call should NOT trigger verification again...")
        response = client.get_completion(
            prompt="What is the capital of France?",
            system_prompt="Answer in one word only.",
            max_tokens=10
        )
        print(f"Response: {response}")
        
        # Test 3: Force reverification
        print(f"\n{Fore.YELLOW}Test 3: Forcing reverification{Style.RESET_ALL}")
        print("Manually setting verified=False to trigger verification again...")
        client.verified = False
        response = client.get_completion(
            prompt="What is 2+2?",
            system_prompt="Answer with just the number.",
            max_tokens=10
        )
        print(f"Response: {response}")
        
        # Test 4: Simulate verification failure
        print(f"\n{Fore.YELLOW}Test 4: Simulating verification failure{Style.RESET_ALL}")
        print("This will demonstrate what happens when verification fails...")
        
        # Override the internal verification method for testing
        original_verify = client.verify_model_accuracy
        
        def mock_verify(*args, **kwargs):
            print(f"\n[!] {client.provider.upper()} {client.model} VERIFICATION FAILED (SIMULATED)")
            print("Question: In what day is Christmas?")
            print("Response: Christmas is in January (simulated incorrect response)")
            print("ALERT: Model did not answer 'December 25th' or similar")
            client.verified = False
            return False
        
        # Replace with mock implementation
        client.verify_model_accuracy = mock_verify
        
        # Force a new verification that will now fail
        client.verified = False
        
        # This will ask the user if they want to continue despite verification failure
        print("You will now be asked if you want to continue despite verification failure.")
        print("Enter 'y' to continue or 'n' to abort.")
        
        try:
            response = client.get_completion(
                prompt="What is the capital of Italy?",
                system_prompt="Answer in one word only.",
                max_tokens=10
            )
            print(f"Response: {response}")
        except RuntimeError as e:
            print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
            print("Test successful - verification failure was handled properly.")
        
        # Restore original verification method
        client.verify_model_accuracy = original_verify
        client.verified = True
        
        # Test 5: Show usage statistics
        print(f"\n{Fore.YELLOW}Test 5: Show usage statistics{Style.RESET_ALL}")
        stats = client.get_usage_statistics()
        print(stats["summary"])
        
    except Exception as e:
        print(f"{Fore.RED}Error during {provider} tests: {e}{Style.RESET_ALL}")

if __name__ == "__main__":
    main()
    print(f"\n{Fore.GREEN}All tests completed.{Style.RESET_ALL}")