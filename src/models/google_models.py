"""Google Gemini model information and utilities."""

import os
import requests

# Dictionary of model information
GEMINI_MODELS = {
    "gemini-2.5-pro-preview": {
        "description": "most powerful, 8M context",
        "max_tokens": 1048576,  # 1M tokens
        "price_per_1k_input": 0.0035,
        "price_per_1k_output": 0.0035
    },
    "gemini-2.5-flash-preview": {
        "description": "efficient, 1M context",
        "max_tokens": 131072,  # 128K tokens
        "price_per_1k_input": 0.00035,
        "price_per_1k_output": 0.00035
    },
    "gemini-1.5-pro": {
        "description": "powerful, 1M context",
        "max_tokens": 131072,  # 128K tokens
        "price_per_1k_input": 0.0035,
        "price_per_1k_output": 0.0035
    },
    "gemini-1.5-flash": {
        "description": "fast, 1M context",
        "max_tokens": 131072,  # 128K tokens
        "price_per_1k_input": 0.00035,
        "price_per_1k_output": 0.00035
    },
    "gemini-2.0-flash": {
        "description": "powerful",
        "max_tokens": 131072,  # 128K tokens
        "price_per_1k_input": 0.00035,
        "price_per_1k_output": 0.00035
    },
    "gemini-2.0-flash-lite": {
        "description": "faster",
        "max_tokens": 32768,  # 32K tokens
        "price_per_1k_input": 0.00025,
        "price_per_1k_output": 0.00025
    }
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

def get_max_tokens_for_model(model_name):
    """Get maximum output tokens for a model."""
    model_info = get_gemini_model_info(model_name)
    return model_info.get("max_tokens", 4096) if model_info else 4096

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
