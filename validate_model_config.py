#!/usr/bin/env python3
"""
Model Configuration Validator

This script validates that the model configurations in the system
match the currently available models from the API providers.
"""

import os
import sys
from dotenv import load_dotenv
from colorama import Fore, Style, init as colorama_init

# Initialize colorama
colorama_init()

# Load environment variables
load_dotenv()

def validate_anthropic_models():
    """Validate Anthropic Claude models."""
    print(f"\n{Fore.CYAN}Validating Anthropic Claude Models:{Style.RESET_ALL}")
    
    try:
        import anthropic
        from src.models.anthropic_models import CLAUDE_MODELS
        
        # Get configured models
        configured_models = set(CLAUDE_MODELS.keys())
        print(f"{Fore.BLUE}Models configured in system: {len(configured_models)}{Style.RESET_ALL}")
        
        # Get available models from API
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        api_models = client.models.list()
        available_models = set(model.id for model in api_models)
        print(f"{Fore.BLUE}Models available from API: {len(available_models)}{Style.RESET_ALL}")
        
        # Check which configured models are available
        valid_models = configured_models.intersection(available_models)
        missing_models = configured_models - available_models
        extra_models = available_models - configured_models
        
        # Print results
        print(f"\n{Fore.GREEN}Valid configured models: {len(valid_models)}{Style.RESET_ALL}")
        for model in sorted(valid_models):
            print(f"  - {model}")
        
        if missing_models:
            print(f"\n{Fore.RED}Configured models not available in API: {len(missing_models)}{Style.RESET_ALL}")
            for model in sorted(missing_models):
                print(f"  - {model}")
        
        if extra_models:
            print(f"\n{Fore.YELLOW}API models not configured in system: {len(extra_models)}{Style.RESET_ALL}")
            for model in sorted(extra_models):
                print(f"  - {model}")
        
        return len(valid_models) > 0
    except Exception as e:
        print(f"{Fore.RED}Error validating Anthropic models: {str(e)}{Style.RESET_ALL}")
        return False

def validate_openai_models():
    """Validate OpenAI models."""
    print(f"\n{Fore.CYAN}Validating OpenAI Models:{Style.RESET_ALL}")
    
    try:
        import openai
        from src.models.openai_models import OPENAI_MODELS
        
        # Get configured models
        configured_models = set(OPENAI_MODELS.keys())
        print(f"{Fore.BLUE}Models configured in system: {len(configured_models)}{Style.RESET_ALL}")
        
        # Get available models from API
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        api_models = client.models.list()
        available_models = set(model.id for model in api_models.data)
        print(f"{Fore.BLUE}Models available from API: {len(available_models)}{Style.RESET_ALL}")
        
        # Check which configured models are available
        valid_models = configured_models.intersection(available_models)
        missing_models = configured_models - available_models
        
        # Only show relevant API models (GPT and O-series)
        relevant_extra_models = {
            model for model in available_models - configured_models
            if (model.startswith('gpt-4') or model.startswith('o') and len(model) < 10)
        }
        
        # Print results
        print(f"\n{Fore.GREEN}Valid configured models: {len(valid_models)}{Style.RESET_ALL}")
        for model in sorted(valid_models):
            print(f"  - {model}")
        
        if missing_models:
            print(f"\n{Fore.RED}Configured models not available in API: {len(missing_models)}{Style.RESET_ALL}")
            for model in sorted(missing_models):
                print(f"  - {model}")
        
        if relevant_extra_models:
            print(f"\n{Fore.YELLOW}Relevant API models not configured in system: {len(relevant_extra_models)}{Style.RESET_ALL}")
            for model in sorted(relevant_extra_models):
                print(f"  - {model}")
        
        return len(valid_models) > 0
    except Exception as e:
        print(f"{Fore.RED}Error validating OpenAI models: {str(e)}{Style.RESET_ALL}")
        return False

def validate_gemini_models():
    """Validate Google Gemini models."""
    print(f"\n{Fore.CYAN}Validating Google Gemini Models:{Style.RESET_ALL}")
    
    try:
        import google.generativeai as genai
        from src.models.google_models import GEMINI_MODELS
        
        # Get configured models
        configured_models = set(GEMINI_MODELS.keys())
        print(f"{Fore.BLUE}Models configured in system: {len(configured_models)}{Style.RESET_ALL}")
        
        # Get available models from API
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        api_models = genai.list_models()
        
        # Gemini API returns models with 'models/' prefix
        available_models = set()
        available_model_full_names = {}
        
        for model in api_models:
            if 'gemini' in model.name.lower():
                # Extract the model name without the 'models/' prefix
                name = model.name.replace('models/', '')
                # Store mapping of short name to full name
                
                # Extract base model name without version
                base_name = name
                if any(x in name for x in ['-001', '-002', '-latest', '-preview-']):
                    for suffix in ['-001', '-002', '-latest', '-preview-', '-exp-']:
                        if suffix in name:
                            base_name = name.split(suffix)[0]
                            break
                
                available_models.add(base_name)
                available_model_full_names[base_name] = model.name
        
        print(f"{Fore.BLUE}Base models available from API: {len(available_models)}{Style.RESET_ALL}")
        
        # Check which configured models have a corresponding API model
        valid_models = set()
        missing_models = set()
        
        for model in configured_models:
            found = False
            for api_model in available_models:
                if model == api_model or api_model.startswith(model):
                    valid_models.add(model)
                    found = True
                    break
            
            if not found:
                missing_models.add(model)
        
        extra_models = available_models - {m for m in valid_models}
        
        # Print results
        print(f"\n{Fore.GREEN}Valid configured models: {len(valid_models)}{Style.RESET_ALL}")
        for model in sorted(valid_models):
            matching_api_models = [api for api in available_model_full_names.keys() 
                                 if api == model or api.startswith(model)]
            
            if matching_api_models:
                api_name = available_model_full_names.get(matching_api_models[0], "Unknown")
                print(f"  - {model} → {api_name}")
            else:
                print(f"  - {model}")
        
        if missing_models:
            print(f"\n{Fore.RED}Configured models not available in API: {len(missing_models)}{Style.RESET_ALL}")
            for model in sorted(missing_models):
                print(f"  - {model}")
        
        if extra_models:
            print(f"\n{Fore.YELLOW}API models not configured in system: {len(extra_models)}{Style.RESET_ALL}")
            for model in sorted(extra_models):
                print(f"  - {model} ({available_model_full_names.get(model, 'Unknown')})")
        
        return len(valid_models) > 0
    except Exception as e:
        print(f"{Fore.RED}Error validating Gemini models: {str(e)}{Style.RESET_ALL}")
        return False

def main():
    """Run the model validation."""
    print(f"{Fore.CYAN}{'=' * 70}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Model Configuration Validator{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 70}{Style.RESET_ALL}")
    
    results = []
    
    # Validate Anthropic models
    anthropic_result = validate_anthropic_models()
    results.append(("Anthropic Claude", anthropic_result))
    
    # Validate OpenAI models
    openai_result = validate_openai_models()
    results.append(("OpenAI", openai_result))
    
    # Validate Google models
    gemini_result = validate_gemini_models()
    results.append(("Google Gemini", gemini_result))
    
    # Print summary
    print(f"\n{Fore.CYAN}{'=' * 70}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Validation Results Summary:{Style.RESET_ALL}")
    
    for provider, success in results:
        if success:
            print(f"{Fore.GREEN}[✓] {provider}: At least one model is valid{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}[✗] {provider}: No valid models found{Style.RESET_ALL}")
    
    # Overall result
    if all(result[1] for result in results):
        print(f"\n{Fore.GREEN}All providers have valid model configurations!{Style.RESET_ALL}")
    else:
        print(f"\n{Fore.YELLOW}Some providers have configuration issues. See details above.{Style.RESET_ALL}")
        print(f"Consider updating the model configuration files in src/models/")

if __name__ == "__main__":
    main()