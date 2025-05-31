"""Anthropic Claude model information and utilities."""

import os
import requests

# Dictionary of model information
CLAUDE_MODELS = {
    "claude-2.0": {
        "description": "standard",
        "max_tokens": 4096,  # Maximum API limit
        "safe_tokens": 3686,  # 90% of max for safety
        "max_safe_tokens": 4055,  # 99% of max for when needed
        "price_per_1k_input": 0.003,
        "price_per_1k_output": 0.015
    },
    "claude-2.1": {
        "description": "standard",
        "max_tokens": 4096,  # Maximum API limit
        "safe_tokens": 3686,  # 90% of max for safety
        "max_safe_tokens": 4055,  # 99% of max for when needed
        "price_per_1k_input": 0.003,
        "price_per_1k_output": 0.015
    },
    "claude-3-5-haiku-20241022": {
        "description": "fast",
        "max_tokens": 4096,  # Maximum API limit
        "safe_tokens": 3686,  # 90% of max for safety
        "max_safe_tokens": 4055,  # 99% of max for when needed
        "price_per_1k_input": 0.00025,
        "price_per_1k_output": 0.00125
    },
    "claude-3-5-sonnet-20240620": {
        "description": "balanced",
        "max_tokens": 8192,  # Maximum API limit
        "safe_tokens": 7372,  # 90% of max for safety
        "max_safe_tokens": 8110,  # 99% of max for when needed
        "price_per_1k_input": 0.0025,
        "price_per_1k_output": 0.0125
    },
    "claude-3-5-sonnet-20241022": {
        "description": "balanced",
        "max_tokens": 8192,  # Maximum API limit
        "safe_tokens": 7372,  # 90% of max for safety
        "max_safe_tokens": 8110,  # 99% of max for when needed
        "price_per_1k_input": 0.0025,
        "price_per_1k_output": 0.0125
    },
    "claude-3-7-sonnet-20250219": {
        "description": "powerful",
        "max_tokens": 16384,  # Maximum API limit
        "safe_tokens": 14745,  # 90% of max for safety
        "max_safe_tokens": 16220,  # 99% of max for when needed
        "price_per_1k_input": 0.003,
        "price_per_1k_output": 0.015
    },
    "claude-3-haiku-20240307": {
        "description": "fastest & cheapest",
        "max_tokens": 4096,  # Maximum API limit
        "safe_tokens": 3686,  # 90% of max for safety
        "max_safe_tokens": 4055,  # 99% of max for when needed
        "price_per_1k_input": 0.00025,
        "price_per_1k_output": 0.00125
    },
    "claude-3-opus-20240229": {
        "description": "most powerful",
        "max_tokens": 32768,  # Maximum API limit
        "safe_tokens": 29491,  # 90% of max for safety
        "max_safe_tokens": 32440,  # 99% of max for when needed
        "price_per_1k_input": 0.015,
        "price_per_1k_output": 0.075
    },
    "claude-3-sonnet-20240229": {
        "description": "powerful",
        "max_tokens": 4096,  # Maximum API limit
        "safe_tokens": 3686,  # 90% of max for safety
        "max_safe_tokens": 4055,  # 99% of max for when needed
        "price_per_1k_input": 0.003,
        "price_per_1k_output": 0.015
    },
    "claude-opus-4-20250514": {
        "description": "most powerful",
        "max_tokens": 32000,  # Maximum API limit
        "safe_tokens": 28800,  # 90% of max for safety
        "max_safe_tokens": 31680,  # 99% of max for when needed
        "price_per_1k_input": 0.015,
        "price_per_1k_output": 0.075
    },
    "claude-sonnet-4-20250514": {
        "description": "powerful",
        "max_tokens": 16384,  # Maximum API limit
        "safe_tokens": 14745,  # 90% of max for safety
        "max_safe_tokens": 16220,  # 99% of max for when needed
        "price_per_1k_input": 0.003,
        "price_per_1k_output": 0.015
    },
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

def get_max_tokens_for_model(model_name, use_safe_limit=True, use_max_safe=False):
    """Get token limit for a model based on safety preferences.
    
    Args:
        model_name: Name of the model
        use_safe_limit: Whether to use the 90% safe limit (default: True)
        use_max_safe: Whether to use the 99% max safe limit (default: False)
        
    Returns:
        The appropriate token limit based on safety preferences
    """
    model_info = get_claude_model_info(model_name)
    if not model_info:
        return 4096  # Default fallback
        
    if use_max_safe:
        return model_info.get("max_safe_tokens", model_info.get("max_tokens", 4096))
    elif use_safe_limit:
        return model_info.get("safe_tokens", model_info.get("max_tokens", 4096))
    else:
        return model_info.get("max_tokens", 4096)

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
