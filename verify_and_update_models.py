#!/usr/bin/env python3
"""
Model Verification and Update Script

This script checks all available models from each provider (Anthropic, OpenAI, Google),
validates them against our configuration, and optionally updates the configuration files.

Run this script every two weeks to ensure model configurations stay current.
"""

import os
import sys
import json
import datetime
import re
from typing import Dict, List, Set, Tuple, Any
from dotenv import load_dotenv
from colorama import Fore, Style, init as colorama_init

# Initialize colorama
colorama_init()

# Load environment variables
load_dotenv()

# Constants
MODEL_FILES = {
    "anthropic": "/home/dpolonia/20250530Computers/src/models/anthropic_models.py",
    "openai": "/home/dpolonia/20250530Computers/src/models/openai_models.py",
    "google": "/home/dpolonia/20250530Computers/src/models/google_models.py"
}

def get_anthropic_models() -> Tuple[List[Dict[str, Any]], Set[str]]:
    """Get all available Anthropic Claude models."""
    try:
        import anthropic
        
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        api_models = client.models.list()
        
        models_info = []
        model_ids = set()
        
        for model in api_models:
            model_id = model.id
            model_ids.add(model_id)
            
            # Get model details
            # Note: We can't directly get max_tokens from the API, so we'll use known values
            if "opus" in model_id:
                max_tokens = 32768
            elif "sonnet" in model_id and "3-5" in model_id:
                max_tokens = 8192
            elif "sonnet" in model_id:
                max_tokens = 16384
            elif "haiku" in model_id:
                max_tokens = 4096
            else:
                max_tokens = 4096  # Default fallback
            
            # Set pricing based on model series
            if "opus" in model_id:
                price_per_1k_input = 0.015
                price_per_1k_output = 0.075
            elif "sonnet" in model_id and "3-5" in model_id:
                price_per_1k_input = 0.0025
                price_per_1k_output = 0.0125
            elif "sonnet" in model_id:
                price_per_1k_input = 0.003
                price_per_1k_output = 0.015
            elif "haiku" in model_id:
                price_per_1k_input = 0.00025
                price_per_1k_output = 0.00125
            else:
                price_per_1k_input = 0.003
                price_per_1k_output = 0.015
            
            # Determine description
            if "opus" in model_id:
                description = "most powerful"
            elif "sonnet" in model_id and "3-5" in model_id:
                description = "balanced"
            elif "sonnet" in model_id:
                description = "powerful"
            elif "haiku" in model_id:
                description = "fast"
            else:
                description = "standard"
            
            # Calculate safe token limits
            safe_tokens = int(max_tokens * 0.9)
            max_safe_tokens = int(max_tokens * 0.99)
            
            models_info.append({
                "id": model_id,
                "max_tokens": max_tokens,
                "safe_tokens": safe_tokens,
                "max_safe_tokens": max_safe_tokens,
                "description": description,
                "price_per_1k_input": price_per_1k_input,
                "price_per_1k_output": price_per_1k_output
            })
        
        return models_info, model_ids
    except Exception as e:
        print(f"{Fore.RED}Error getting Anthropic models: {str(e)}{Style.RESET_ALL}")
        return [], set()

def get_openai_models() -> Tuple[List[Dict[str, Any]], Set[str]]:
    """Get all available OpenAI models."""
    try:
        import openai
        
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        api_models = client.models.list()
        
        models_info = []
        model_ids = set()
        
        for model in api_models.data:
            model_id = model.id
            
            # Only include GPT and O-series models
            if not (model_id.startswith("gpt-") or (model_id.startswith("o") and len(model_id) < 10)):
                continue
            
            model_ids.add(model_id)
            
            # Determine max tokens based on model name
            if "32k" in model_id:
                max_tokens = 32768
            elif "16k" in model_id:
                max_tokens = 16384
            elif "gpt-4" in model_id or model_id.startswith("o"):
                max_tokens = 16384
            elif "gpt-3.5" in model_id:
                max_tokens = 4096
            else:
                max_tokens = 4096  # Default fallback
            
            # Set pricing based on model series
            if "gpt-4.5" in model_id:
                price_per_1k_input = 0.01
                price_per_1k_output = 0.03
            elif "gpt-4o" in model_id:
                price_per_1k_input = 0.005
                price_per_1k_output = 0.015
            elif "gpt-4o-mini" in model_id:
                price_per_1k_input = 0.0015
                price_per_1k_output = 0.006
            elif model_id in ["o1", "o3"]:
                price_per_1k_input = 0.015
                price_per_1k_output = 0.075
            elif model_id in ["o4-mini"]:
                price_per_1k_input = 0.005
                price_per_1k_output = 0.015
            else:
                price_per_1k_input = 0.005
                price_per_1k_output = 0.015
            
            # Determine description
            if "gpt-4.5" in model_id:
                description = "most powerful"
            elif "gpt-4o" in model_id and "mini" not in model_id:
                description = "powerful"
            elif "gpt-4o-mini" in model_id:
                description = "balanced"
            elif model_id in ["o1", "o3"]:
                description = "powerful reasoning"
            elif model_id in ["o4-mini"]:
                description = "fast reasoning"
            else:
                description = "standard"
            
            # Calculate safe token limits
            safe_tokens = int(max_tokens * 0.9)
            max_safe_tokens = int(max_tokens * 0.99)
            
            models_info.append({
                "id": model_id,
                "max_tokens": max_tokens,
                "safe_tokens": safe_tokens,
                "max_safe_tokens": max_safe_tokens,
                "description": description,
                "price_per_1k_input": price_per_1k_input,
                "price_per_1k_output": price_per_1k_output
            })
        
        return models_info, model_ids
    except Exception as e:
        print(f"{Fore.RED}Error getting OpenAI models: {str(e)}{Style.RESET_ALL}")
        return [], set()

def get_gemini_models() -> Tuple[List[Dict[str, Any]], Set[str]]:
    """Get all available Google Gemini models."""
    try:
        import google.generativeai as genai
        
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        api_models = genai.list_models()
        
        models_info = []
        full_model_ids = set()
        simplified_model_ids = set()
        
        for model in api_models:
            if 'gemini' not in model.name.lower():
                continue
                
            full_model_id = model.name
            full_model_ids.add(full_model_id)
            
            # Extract the model name without the 'models/' prefix
            model_id = full_model_id.replace('models/', '')
            
            # For simplified model IDs, remove version numbers and suffixes
            simplified_id = model_id
            if any(x in model_id for x in ['-001', '-002', '-latest', '-preview-']):
                for suffix in ['-001', '-002', '-latest', '-preview-', '-exp-']:
                    if suffix in model_id:
                        simplified_id = model_id.split(suffix)[0]
                        break
            
            simplified_model_ids.add(simplified_id)
            
            # Determine max tokens based on model name
            if "pro" in model_id and "1.5" in model_id:
                max_tokens = 131072  # 128K tokens
            elif "flash" in model_id and "1.5" in model_id:
                max_tokens = 131072  # 128K tokens
            elif "2.5" in model_id and "pro" in model_id:
                max_tokens = 1048576  # 1M tokens
            elif "2.5" in model_id and "flash" in model_id:
                max_tokens = 131072  # 128K tokens
            elif "2.0" in model_id and "flash-lite" in model_id:
                max_tokens = 32768  # 32K tokens
            elif "2.0" in model_id:
                max_tokens = 131072  # 128K tokens
            else:
                max_tokens = 32768  # Default fallback
            
            # Set pricing based on model series
            if "pro" in model_id:
                price_per_1k_input = 0.0035
                price_per_1k_output = 0.0035
            elif "flash" in model_id:
                price_per_1k_input = 0.00035
                price_per_1k_output = 0.00035
            else:
                price_per_1k_input = 0.0035
                price_per_1k_output = 0.0035
            
            # Determine description
            if "2.5" in model_id and "pro" in model_id:
                description = "most powerful, 8M context"
            elif "2.5" in model_id and "flash" in model_id:
                description = "efficient, 1M context"
            elif "pro" in model_id and "1.5" in model_id:
                description = "powerful, 1M context"
            elif "flash" in model_id and "1.5" in model_id:
                description = "fast, 1M context"
            elif "2.0" in model_id and "flash" in model_id:
                description = "powerful"
            elif "2.0" in model_id and "flash-lite" in model_id:
                description = "faster"
            else:
                description = "standard"
            
            # Calculate safe token limits
            safe_tokens = int(max_tokens * 0.9)
            max_safe_tokens = int(max_tokens * 0.99)
            
            # Store both the simplified ID and the full model ID
            models_info.append({
                "id": simplified_id,
                "full_id": full_model_id,
                "max_tokens": max_tokens,
                "safe_tokens": safe_tokens,
                "max_safe_tokens": max_safe_tokens,
                "description": description,
                "price_per_1k_input": price_per_1k_input,
                "price_per_1k_output": price_per_1k_output
            })
        
        return models_info, simplified_model_ids
    except Exception as e:
        print(f"{Fore.RED}Error getting Gemini models: {str(e)}{Style.RESET_ALL}")
        return [], set()

def parse_model_config(file_path: str) -> Dict[str, Dict[str, Any]]:
    """Parse model configuration from a Python file."""
    try:
        # Read the file content
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Extract the dictionary definition using regex
        if "CLAUDE_MODELS" in content:
            dict_name = "CLAUDE_MODELS"
        elif "OPENAI_MODELS" in content:
            dict_name = "OPENAI_MODELS"
        elif "GEMINI_MODELS" in content:
            dict_name = "GEMINI_MODELS"
        else:
            return {}
        
        # Extract the dictionary content using a simple approach
        dict_start = content.find(f"{dict_name} = {{")
        if dict_start == -1:
            return {}
        
        dict_start = content.find("{", dict_start)
        bracket_count = 1
        dict_end = dict_start + 1
        
        while bracket_count > 0 and dict_end < len(content):
            if content[dict_end] == "{":
                bracket_count += 1
            elif content[dict_end] == "}":
                bracket_count -= 1
            dict_end += 1
        
        if bracket_count != 0:
            return {}
        
        dict_str = content[dict_start:dict_end]
        
        # Convert Python-style dict to JSON-style
        dict_str = re.sub(r'#.*$', '', dict_str, flags=re.MULTILINE)  # Remove comments
        dict_str = re.sub(r',\s*}', '}', dict_str)  # Remove trailing commas
        dict_str = re.sub(r'\'', '"', dict_str)  # Convert single quotes to double quotes
        dict_str = re.sub(r',\s*$', '', dict_str, flags=re.MULTILINE)  # Remove trailing commas at end of lines
        
        # Parse models from dict_str
        models = {}
        model_pattern = r'"([^"]+)":\s*{([^{}]+)}'
        for match in re.finditer(model_pattern, dict_str):
            model_id = match.group(1)
            model_props = match.group(2)
            
            props = {}
            
            # Parse description
            desc_match = re.search(r'"description":\s*"([^"]+)"', model_props)
            if desc_match:
                props["description"] = desc_match.group(1)
            
            # Parse max_tokens
            tokens_match = re.search(r'"max_tokens":\s*(\d+)', model_props)
            if tokens_match:
                props["max_tokens"] = int(tokens_match.group(1))
            
            # Parse safe_tokens
            safe_tokens_match = re.search(r'"safe_tokens":\s*(\d+)', model_props)
            if safe_tokens_match:
                props["safe_tokens"] = int(safe_tokens_match.group(1))
            
            # Parse max_safe_tokens
            max_safe_tokens_match = re.search(r'"max_safe_tokens":\s*(\d+)', model_props)
            if max_safe_tokens_match:
                props["max_safe_tokens"] = int(max_safe_tokens_match.group(1))
            
            # Parse price_per_1k_input
            price_input_match = re.search(r'"price_per_1k_input":\s*([\d\.]+)', model_props)
            if price_input_match:
                props["price_per_1k_input"] = float(price_input_match.group(1))
            
            # Parse price_per_1k_output
            price_output_match = re.search(r'"price_per_1k_output":\s*([\d\.]+)', model_props)
            if price_output_match:
                props["price_per_1k_output"] = float(price_output_match.group(1))
            
            models[model_id] = props
        
        return models
    except Exception as e:
        print(f"{Fore.RED}Error parsing model config from {file_path}: {str(e)}{Style.RESET_ALL}")
        return {}

def update_model_config(file_path: str, dict_name: str, models: Dict[str, Dict[str, Any]]) -> bool:
    """Update model configuration in a Python file."""
    try:
        # Read the file content
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Extract the dictionary definition using regex
        dict_start = content.find(f"{dict_name} = {{")
        if dict_start == -1:
            return False
        
        dict_start = content.find("{", dict_start)
        bracket_count = 1
        dict_end = dict_start + 1
        
        while bracket_count > 0 and dict_end < len(content):
            if content[dict_end] == "{":
                bracket_count += 1
            elif content[dict_end] == "}":
                bracket_count -= 1
            dict_end += 1
        
        if bracket_count != 0:
            return False
        
        # Generate new dictionary content
        new_dict = "{\n"
        for model_id, props in sorted(models.items()):
            new_dict += f'    "{model_id}": {{\n'
            
            if "description" in props:
                new_dict += f'        "description": "{props["description"]}",\n'
            
            if "max_tokens" in props:
                new_dict += f'        "max_tokens": {props["max_tokens"]},  # Maximum API limit\n'
            
            if "safe_tokens" in props:
                new_dict += f'        "safe_tokens": {props["safe_tokens"]},  # 90% of max for safety\n'
            
            if "max_safe_tokens" in props:
                new_dict += f'        "max_safe_tokens": {props["max_safe_tokens"]},  # 99% of max for when needed\n'
            
            if "price_per_1k_input" in props:
                new_dict += f'        "price_per_1k_input": {props["price_per_1k_input"]},\n'
            
            if "price_per_1k_output" in props:
                new_dict += f'        "price_per_1k_output": {props["price_per_1k_output"]}\n'
            
            new_dict += "    },\n"
        
        new_dict += "}"
        
        # Replace the old dictionary with the new one
        new_content = content[:dict_start] + new_dict + content[dict_end:]
        
        # Write the updated content back to the file
        with open(file_path, 'w') as f:
            f.write(new_content)
        
        return True
    except Exception as e:
        print(f"{Fore.RED}Error updating model config in {file_path}: {str(e)}{Style.RESET_ALL}")
        return False

def verify_anthropic_models(update: bool = False) -> bool:
    """Verify and optionally update Anthropic Claude models."""
    print(f"\n{Fore.CYAN}Verifying Anthropic Claude Models:{Style.RESET_ALL}")
    
    # Get available models from API
    api_models, api_model_ids = get_anthropic_models()
    
    if not api_models:
        print(f"{Fore.RED}Failed to get Anthropic models from API{Style.RESET_ALL}")
        return False
    
    print(f"{Fore.BLUE}Found {len(api_model_ids)} models available from Anthropic API{Style.RESET_ALL}")
    
    # Parse current model configuration
    file_path = MODEL_FILES["anthropic"]
    configured_models = parse_model_config(file_path)
    
    if not configured_models:
        print(f"{Fore.RED}Failed to parse current Anthropic model configuration{Style.RESET_ALL}")
        return False
    
    print(f"{Fore.BLUE}Found {len(configured_models)} models in current configuration{Style.RESET_ALL}")
    
    # Check for mismatches
    configured_model_ids = set(configured_models.keys())
    missing_models = api_model_ids - configured_model_ids
    extra_models = configured_model_ids - api_model_ids
    
    # Print results
    if not missing_models and not extra_models:
        print(f"{Fore.GREEN}All Anthropic models are up to date!{Style.RESET_ALL}")
        return True
    
    if missing_models:
        print(f"\n{Fore.YELLOW}Models available in API but not configured: {len(missing_models)}{Style.RESET_ALL}")
        for model_id in sorted(missing_models):
            print(f"  - {model_id}")
    
    if extra_models:
        print(f"\n{Fore.YELLOW}Models configured but not available in API: {len(extra_models)}{Style.RESET_ALL}")
        for model_id in sorted(extra_models):
            print(f"  - {model_id}")
    
    # Update configuration if requested
    if update:
        print(f"\n{Fore.CYAN}Updating Anthropic model configuration...{Style.RESET_ALL}")
        
        # Keep existing configured models (they might still work)
        updated_models = configured_models.copy()
        
        # Add missing models
        for model_info in api_models:
            model_id = model_info["id"]
            if model_id not in updated_models:
                updated_models[model_id] = {
                    "description": model_info["description"],
                    "max_tokens": model_info["max_tokens"],
                    "safe_tokens": model_info["safe_tokens"],
                    "max_safe_tokens": model_info["max_safe_tokens"],
                    "price_per_1k_input": model_info["price_per_1k_input"],
                    "price_per_1k_output": model_info["price_per_1k_output"]
                }
        
        # Update the configuration file
        if update_model_config(file_path, "CLAUDE_MODELS", updated_models):
            print(f"{Fore.GREEN}Successfully updated Anthropic model configuration!{Style.RESET_ALL}")
            return True
        else:
            print(f"{Fore.RED}Failed to update Anthropic model configuration{Style.RESET_ALL}")
            return False
    
    return True

def verify_openai_models(update: bool = False) -> bool:
    """Verify and optionally update OpenAI models."""
    print(f"\n{Fore.CYAN}Verifying OpenAI Models:{Style.RESET_ALL}")
    
    # Get available models from API
    api_models, api_model_ids = get_openai_models()
    
    if not api_models:
        print(f"{Fore.RED}Failed to get OpenAI models from API{Style.RESET_ALL}")
        return False
    
    print(f"{Fore.BLUE}Found {len(api_model_ids)} relevant models available from OpenAI API{Style.RESET_ALL}")
    
    # Parse current model configuration
    file_path = MODEL_FILES["openai"]
    configured_models = parse_model_config(file_path)
    
    if not configured_models:
        print(f"{Fore.RED}Failed to parse current OpenAI model configuration{Style.RESET_ALL}")
        return False
    
    print(f"{Fore.BLUE}Found {len(configured_models)} models in current configuration{Style.RESET_ALL}")
    
    # Check for mismatches
    configured_model_ids = set(configured_models.keys())
    missing_models = api_model_ids - configured_model_ids
    extra_models = configured_model_ids - api_model_ids
    
    # Filter missing models to only include relevant ones
    relevant_missing_models = {model_id for model_id in missing_models 
                               if (model_id.startswith("gpt-4") or 
                                   model_id.startswith("o") and len(model_id) < 10)}
    
    # Print results
    if not relevant_missing_models and not extra_models:
        print(f"{Fore.GREEN}All important OpenAI models are up to date!{Style.RESET_ALL}")
        return True
    
    if relevant_missing_models:
        print(f"\n{Fore.YELLOW}Relevant models available in API but not configured: {len(relevant_missing_models)}{Style.RESET_ALL}")
        for model_id in sorted(relevant_missing_models):
            print(f"  - {model_id}")
    
    if extra_models:
        print(f"\n{Fore.YELLOW}Models configured but not available in API: {len(extra_models)}{Style.RESET_ALL}")
        for model_id in sorted(extra_models):
            print(f"  - {model_id}")
    
    # Update configuration if requested
    if update:
        print(f"\n{Fore.CYAN}Updating OpenAI model configuration...{Style.RESET_ALL}")
        
        # Keep existing configured models (they might still work)
        updated_models = configured_models.copy()
        
        # Add missing models (only add important ones)
        for model_info in api_models:
            model_id = model_info["id"]
            if model_id in relevant_missing_models:
                updated_models[model_id] = {
                    "description": model_info["description"],
                    "max_tokens": model_info["max_tokens"],
                    "safe_tokens": model_info["safe_tokens"],
                    "max_safe_tokens": model_info["max_safe_tokens"],
                    "price_per_1k_input": model_info["price_per_1k_input"],
                    "price_per_1k_output": model_info["price_per_1k_output"]
                }
        
        # Update the configuration file
        if update_model_config(file_path, "OPENAI_MODELS", updated_models):
            print(f"{Fore.GREEN}Successfully updated OpenAI model configuration!{Style.RESET_ALL}")
            return True
        else:
            print(f"{Fore.RED}Failed to update OpenAI model configuration{Style.RESET_ALL}")
            return False
    
    return True

def verify_gemini_models(update: bool = False) -> bool:
    """Verify and optionally update Google Gemini models."""
    print(f"\n{Fore.CYAN}Verifying Google Gemini Models:{Style.RESET_ALL}")
    
    # Get available models from API
    api_models, api_model_ids = get_gemini_models()
    
    if not api_models:
        print(f"{Fore.RED}Failed to get Gemini models from API{Style.RESET_ALL}")
        return False
    
    print(f"{Fore.BLUE}Found {len(api_model_ids)} base models available from Gemini API{Style.RESET_ALL}")
    
    # Parse current model configuration
    file_path = MODEL_FILES["google"]
    configured_models = parse_model_config(file_path)
    
    if not configured_models:
        print(f"{Fore.RED}Failed to parse current Gemini model configuration{Style.RESET_ALL}")
        return False
    
    print(f"{Fore.BLUE}Found {len(configured_models)} models in current configuration{Style.RESET_ALL}")
    
    # Check for mismatches
    configured_model_ids = set(configured_models.keys())
    missing_models = api_model_ids - configured_model_ids
    extra_models = configured_model_ids - api_model_ids
    
    # Map from simplified model IDs to full model IDs
    model_id_map = {model["id"]: model["full_id"] for model in api_models}
    
    # Print results
    valid_models = configured_model_ids.intersection(api_model_ids)
    
    if not missing_models and not extra_models:
        print(f"{Fore.GREEN}All Gemini models are up to date!{Style.RESET_ALL}")
        return True
    
    print(f"\n{Fore.GREEN}Valid configured models: {len(valid_models)}{Style.RESET_ALL}")
    for model_id in sorted(valid_models):
        if model_id in model_id_map:
            print(f"  - {model_id} → {model_id_map[model_id]}")
        else:
            print(f"  - {model_id}")
    
    if missing_models:
        print(f"\n{Fore.YELLOW}Models available in API but not configured: {len(missing_models)}{Style.RESET_ALL}")
        for model_id in sorted(missing_models):
            if model_id in model_id_map:
                print(f"  - {model_id} ({model_id_map[model_id]})")
            else:
                print(f"  - {model_id}")
    
    if extra_models:
        print(f"\n{Fore.YELLOW}Models configured but not available in API: {len(extra_models)}{Style.RESET_ALL}")
        for model_id in sorted(extra_models):
            print(f"  - {model_id}")
    
    # Update configuration if requested
    if update:
        print(f"\n{Fore.CYAN}Updating Gemini model configuration...{Style.RESET_ALL}")
        
        # Start with existing valid models
        updated_models = {model_id: configured_models[model_id] for model_id in valid_models}
        
        # Add missing models (prioritize stable versions)
        for model_info in api_models:
            model_id = model_info["id"]
            if model_id not in updated_models and "exp" not in model_id and "preview" not in model_id:
                updated_models[model_id] = {
                    "description": model_info["description"],
                    "max_tokens": model_info["max_tokens"],
                    "safe_tokens": model_info["safe_tokens"],
                    "max_safe_tokens": model_info["max_safe_tokens"],
                    "price_per_1k_input": model_info["price_per_1k_input"],
                    "price_per_1k_output": model_info["price_per_1k_output"]
                }
        
        # Update the configuration file
        if update_model_config(file_path, "GEMINI_MODELS", updated_models):
            print(f"{Fore.GREEN}Successfully updated Gemini model configuration!{Style.RESET_ALL}")
            return True
        else:
            print(f"{Fore.RED}Failed to update Gemini model configuration{Style.RESET_ALL}")
            return False
    
    return True

def check_for_updates():
    """Check for major new model releases or EOL announcements."""
    print(f"\n{Fore.CYAN}Checking for major model updates and announcements:{Style.RESET_ALL}")
    
    # Anthropic announcements
    print(f"\n{Fore.YELLOW}Anthropic:{Style.RESET_ALL}")
    print("- Visit https://docs.anthropic.com/claude/docs/models-overview for latest model information")
    print("- Check https://www.anthropic.com/news for major announcements")
    
    # OpenAI announcements
    print(f"\n{Fore.YELLOW}OpenAI:{Style.RESET_ALL}")
    print("- Visit https://platform.openai.com/docs/models for latest model information")
    print("- Check https://openai.com/blog for major announcements")
    
    # Google announcements
    print(f"\n{Fore.YELLOW}Google:{Style.RESET_ALL}")
    print("- Visit https://ai.google.dev/models/gemini for latest model information")
    print("- Check https://blog.google/technology/ai/ for major announcements")

def main():
    """Run the model verification and update script."""
    # Print header
    now = datetime.datetime.now()
    print(f"{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Model Verification and Update Report - {now.strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}")
    
    # Check for --update flag
    update = "--update" in sys.argv
    
    if update:
        print(f"{Fore.YELLOW}Update mode enabled - will update model configurations if needed{Style.RESET_ALL}")
    else:
        print(f"{Fore.BLUE}Verification mode - will only check models (use --update to update configs){Style.RESET_ALL}")
    
    # Verify models for each provider
    results = []
    
    # Anthropic models
    try:
        anthropic_result = verify_anthropic_models(update)
        results.append(("Anthropic Claude", anthropic_result))
    except Exception as e:
        print(f"{Fore.RED}Error verifying Anthropic models: {str(e)}{Style.RESET_ALL}")
        results.append(("Anthropic Claude", False))
    
    # OpenAI models
    try:
        openai_result = verify_openai_models(update)
        results.append(("OpenAI", openai_result))
    except Exception as e:
        print(f"{Fore.RED}Error verifying OpenAI models: {str(e)}{Style.RESET_ALL}")
        results.append(("OpenAI", False))
    
    # Gemini models
    try:
        gemini_result = verify_gemini_models(update)
        results.append(("Google Gemini", gemini_result))
    except Exception as e:
        print(f"{Fore.RED}Error verifying Gemini models: {str(e)}{Style.RESET_ALL}")
        results.append(("Google Gemini", False))
    
    # Check for major updates
    check_for_updates()
    
    # Print summary
    print(f"\n{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Verification Results Summary:{Style.RESET_ALL}")
    
    for provider, success in results:
        if success:
            print(f"{Fore.GREEN}[✓] {provider}: Verification completed successfully{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}[✗] {provider}: Verification failed{Style.RESET_ALL}")
    
    # Schedule reminder
    next_check = now + datetime.timedelta(days=14)
    print(f"\n{Fore.YELLOW}Next verification should be run on: {next_check.strftime('%Y-%m-%d')}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}To schedule, add this to your calendar or set up a recurring cron job:{Style.RESET_ALL}")
    print(f"  0 9 {next_check.day} {next_check.month} * python /home/dpolonia/20250530Computers/verify_and_update_models.py")
    
    # Instructions for updating
    if not update:
        print(f"\n{Fore.CYAN}To update model configurations, run:{Style.RESET_ALL}")
        print(f"  python /home/dpolonia/20250530Computers/verify_and_update_models.py --update")

if __name__ == "__main__":
    main()