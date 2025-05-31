"""Google Gemini model information and utilities."""

import os
import requests

# Dictionary of model information
GEMINI_MODELS = {
    "gemini-1.0-pro-vision": {
        "description": "standard",
        "max_tokens": 32768,  # Maximum API limit
        "safe_tokens": 29491,  # 90% of max for safety
        "max_safe_tokens": 32440,  # 99% of max for when needed
        "price_per_1k_input": 0.0035,
        "price_per_1k_output": 0.0035
    },
    "gemini-1.5-flash": {
        "description": "fast, 1M context",
        "max_tokens": 131072,  # Maximum API limit
        "safe_tokens": 117964,  # 90% of max for safety
        "max_safe_tokens": 129761,  # 99% of max for when needed
        "price_per_1k_input": 0.00035,
        "price_per_1k_output": 0.00035
    },
    "gemini-1.5-flash-8b": {
        "description": "fast, 1M context",
        "max_tokens": 131072,  # Maximum API limit
        "safe_tokens": 117964,  # 90% of max for safety
        "max_safe_tokens": 129761,  # 99% of max for when needed
        "price_per_1k_input": 0.00035,
        "price_per_1k_output": 0.00035
    },
    "gemini-1.5-pro": {
        "description": "powerful, 1M context",
        "max_tokens": 131072,  # Maximum API limit
        "safe_tokens": 117964,  # 90% of max for safety
        "max_safe_tokens": 129761,  # 99% of max for when needed
        "price_per_1k_input": 0.0035,
        "price_per_1k_output": 0.0035
    },
    "gemini-2.0-flash": {
        "description": "powerful",
        "max_tokens": 131072,  # Maximum API limit
        "safe_tokens": 117964,  # 90% of max for safety
        "max_safe_tokens": 129761,  # 99% of max for when needed
        "price_per_1k_input": 0.00035,
        "price_per_1k_output": 0.00035
    },
    "gemini-2.0-flash-lite": {
        "description": "faster",
        "max_tokens": 32768,  # Maximum API limit
        "safe_tokens": 29491,  # 90% of max for safety
        "max_safe_tokens": 32440,  # 99% of max for when needed
        "price_per_1k_input": 0.00025,
        "price_per_1k_output": 0.00025
    },
    "gemini-2.0-flash-live": {
        "description": "powerful",
        "max_tokens": 131072,  # Maximum API limit
        "safe_tokens": 117964,  # 90% of max for safety
        "max_safe_tokens": 129761,  # 99% of max for when needed
        "price_per_1k_input": 0.00035,
        "price_per_1k_output": 0.00035
    },
    "gemini-2.0-pro-exp": {
        "description": "experimental, powerful",
        "max_tokens": 131072,  # Maximum API limit
        "safe_tokens": 117964,  # 90% of max for safety
        "max_safe_tokens": 129761,  # 99% of max for when needed
        "price_per_1k_input": 0.00035,
        "price_per_1k_output": 0.00035
    },
    "gemini-2.5-flash": {
        "description": "efficient, 1M context",
        "max_tokens": 131072,  # Maximum API limit
        "safe_tokens": 117964,  # 90% of max for safety
        "max_safe_tokens": 129761,  # 99% of max for when needed
        "price_per_1k_input": 0.00035,
        "price_per_1k_output": 0.00035
    },
    "gemini-2.5-pro": {
        "description": "most powerful, 8M context",
        "max_tokens": 1048576,  # Maximum API limit
        "safe_tokens": 943718,  # 90% of max for safety
        "max_safe_tokens": 1038090,  # 99% of max for when needed
        "price_per_1k_input": 0.0035,
        "price_per_1k_output": 0.0035
    },
    "gemini-pro-vision": {
        "description": "standard",
        "max_tokens": 32768,  # Maximum API limit
        "safe_tokens": 29491,  # 90% of max for safety
        "max_safe_tokens": 32440,  # 99% of max for when needed
        "price_per_1k_input": 0.0035,
        "price_per_1k_output": 0.0035
    },
}

def get_gemini_model_choices():
    """Return a list of available Gemini models with descriptions."""
    return [
        f"{model} ({info['description']})"
        for model, info in GEMINI_MODELS.items()
    ]

def get_gemini_model_info(model_name):
    """Get information for a specific Gemini model.
    
    Args:
        model_name: The name of the model, may include description
        
    Returns:
        Dict with model information or None if not found
    """
    # Extract base model name without description
    base_model = model_name.split(" (")[0]
    return GEMINI_MODELS.get(base_model)

def get_max_tokens_for_model(model_name, use_safe_limit=True, use_max_safe=False):
    """Get token limit for a model based on safety preferences.
    
    Args:
        model_name: Name of the model
        use_safe_limit: Whether to use the 90% safe limit (default: True)
        use_max_safe: Whether to use the 99% max safe limit (default: False)
        
    Returns:
        The appropriate token limit based on safety preferences
    """
    model_info = get_gemini_model_info(model_name)
    if not model_info:
        return 4096  # Default fallback
        
    if use_max_safe:
        return model_info.get("max_safe_tokens", model_info.get("max_tokens", 4096))
    elif use_safe_limit:
        return model_info.get("safe_tokens", model_info.get("max_tokens", 4096))
    else:
        return model_info.get("max_tokens", 4096)

def validate_api_key():
    """Validate the Google API key."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return False
        
    try:
        response = requests.get(
            f"https://generativelanguage.googleapis.com/v1/models?key={api_key}"
        )
        return response.status_code == 200
    except Exception:
        return False
