"""OpenAI model information and utilities."""

import os
import requests

# Dictionary of model information
OPENAI_MODELS = {
    "gpt-4.5-preview": {
        "description": "most powerful",
        "max_tokens": 16384,
        "price_per_1k_input": 0.01,
        "price_per_1k_output": 0.03
    },
    "gpt-4o": {
        "description": "powerful",
        "max_tokens": 16384,
        "price_per_1k_input": 0.005,
        "price_per_1k_output": 0.015
    },
    "gpt-4o-mini": {
        "description": "balanced",
        "max_tokens": 16384,
        "price_per_1k_input": 0.0015,
        "price_per_1k_output": 0.006
    },
    "o1": {
        "description": "powerful reasoning",
        "max_tokens": 16384,
        "price_per_1k_input": 0.015,
        "price_per_1k_output": 0.075
    },
    "o3": {
        "description": "powerful reasoning",
        "max_tokens": 16384,
        "price_per_1k_input": 0.015,
        "price_per_1k_output": 0.075
    },
    "o4-mini": {
        "description": "fast reasoning",
        "max_tokens": 16384,
        "price_per_1k_input": 0.005,
        "price_per_1k_output": 0.015
    }
}

def get_openai_model_choices():
    """Return a list of available OpenAI models with descriptions."""
    return [
        f"{model} ({info['description']})"
        for model, info in OPENAI_MODELS.items()
    ]

def get_openai_model_info(model_name):
    """Get information for a specific OpenAI model.
    
    Args:
        model_name: The name of the model, may include description
        
    Returns:
        Dict with model information or None if not found
    """
    # Extract base model name without description
    base_model = model_name.split(" (")[0]
    return OPENAI_MODELS.get(base_model)

def get_max_tokens_for_model(model_name):
    """Get maximum output tokens for a model."""
    model_info = get_openai_model_info(model_name)
    return model_info.get("max_tokens", 4096) if model_info else 4096

def validate_api_key():
    """Validate the OpenAI API key."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return False
        
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    
    try:
        response = requests.get("https://api.openai.com/v1/models", headers=headers)
        return response.status_code == 200
    except Exception:
        return False
