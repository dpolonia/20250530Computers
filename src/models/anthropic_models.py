"""Anthropic Claude model information and utilities."""

import os
import requests

# Dictionary of model information
CLAUDE_MODELS = {
    "claude-opus-4-20250514": {
        "description": "most powerful",
        "max_tokens": 200000,
        "price_per_1k_input": 0.015,
        "price_per_1k_output": 0.075
    },
    "claude-sonnet-4-20250514": {
        "description": "powerful",
        "max_tokens": 200000,
        "price_per_1k_input": 0.003,
        "price_per_1k_output": 0.015
    },
    "claude-3-7-sonnet-20250219": {
        "description": "powerful",
        "max_tokens": 200000,
        "price_per_1k_input": 0.003,
        "price_per_1k_output": 0.015
    },
    "claude-3-5-sonnet-20241022": {
        "description": "balanced",
        "max_tokens": 200000,
        "price_per_1k_input": 0.0025,
        "price_per_1k_output": 0.0125
    },
    "claude-3-5-haiku-20241022": {
        "description": "fast",
        "max_tokens": 200000,
        "price_per_1k_input": 0.00025,
        "price_per_1k_output": 0.00125
    },
    "claude-3-haiku-20240307": {
        "description": "fastest & cheapest",
        "max_tokens": 200000,
        "price_per_1k_input": 0.00025,
        "price_per_1k_output": 0.00125
    }
}

def get_claude_model_choices():
    """Return a list of available Claude models with descriptions."""
    return [
        f"{model} ({info['description']})"
        for model, info in CLAUDE_MODELS.items()
    ]

def get_claude_model_info(model_name):
    """Get information for a specific Claude model.
    
    Args:
        model_name: The name of the model, may include description
        
    Returns:
        Dict with model information or None if not found
    """
    # Extract base model name without description
    base_model = model_name.split(" (")[0]
    return CLAUDE_MODELS.get(base_model)

def get_max_tokens_for_model(model_name):
    """Get maximum output tokens for a model."""
    model_info = get_claude_model_info(model_name)
    return model_info.get("max_tokens", 4096) if model_info else 4096

def validate_api_key():
    """Validate the Anthropic API key."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return False
        
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01"
    }
    
    try:
        response = requests.get("https://api.anthropic.com/v1/models", headers=headers)
        return response.status_code == 200
    except Exception:
        return False
