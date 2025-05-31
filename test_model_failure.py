#!/usr/bin/env python3
"""
Model Verification Failure Test Script

Tests the model verification failure handling.
"""

import os
import sys
from dotenv import load_dotenv
from colorama import Fore, Style, init as colorama_init

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import LLM client
from src.utils.llm_client import get_llm_client, BaseLLMClient

# Initialize colorama
colorama_init()

# Load environment variables
load_dotenv()

# Override the verify_model_accuracy method to always fail
original_verify_method = BaseLLMClient.verify_model_accuracy

def force_fail_verification(self):
    """Modified verification method that always fails."""
    response = original_verify_method(self)
    # Override the result to force failure
    self.verified = False
    print(f"\n{Fore.RED}[!] SIMULATED FAILURE: Verification forced to fail for testing{Style.RESET_ALL}")
    return False

# Replace the original method with our forced failure version
BaseLLMClient.verify_model_accuracy = force_fail_verification

def test_model_failure(provider, model_name):
    """Test a model with verification that's forced to fail."""
    print(f"{Fore.CYAN}{'=' * 70}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Testing {provider.upper()} Model: {model_name} (Forced Failure){Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 70}{Style.RESET_ALL}")
    
    try:
        # Initialize client with verification (which will fail)
        client = get_llm_client(provider, model_name, verify=True)
        
        print(f"\n{Fore.RED}[!] Model was initialized despite verification failure!{Style.RESET_ALL}")
        
        # Print usage statistics
        stats = client.get_usage_statistics()
        print(f"\n{Fore.CYAN}Usage Statistics:{Style.RESET_ALL}")
        print(f"Total tokens: {stats['total_tokens']}")
        print(f"Total cost: ${stats['total_cost']:.6f}")
        
    except Exception as e:
        print(f"\n{Fore.YELLOW}Expected error: {str(e)}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}[✓] Failure handling is working properly{Style.RESET_ALL}")

def main():
    """Run model verification failure test."""
    print(f"{Fore.CYAN}{'=' * 70}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Model Verification Failure Test{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 70}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}This test simulates verification failures to test error handling{Style.RESET_ALL}")
    
    # Test with a small model for speed
    test_model_failure("anthropic", "claude-3-haiku-20240307")
    
    print(f"\n{Fore.CYAN}{'=' * 70}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Test Complete{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 70}{Style.RESET_ALL}")
    
    # Restore the original verification method
    BaseLLMClient.verify_model_accuracy = original_verify_method
    print(f"{Fore.GREEN}Original verification method restored{Style.RESET_ALL}")

if __name__ == "__main__":
    main()