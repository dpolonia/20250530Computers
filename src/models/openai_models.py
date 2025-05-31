"""OpenAI model information and utilities."""

import os
import requests

# Dictionary of model information
OPENAI_MODELS = {
    "gpt-4": {
        "description": "standard",
        "max_tokens": 16384,  # Maximum API limit
        "safe_tokens": 14745,  # 90% of max for safety
        "max_safe_tokens": 16220,  # 99% of max for when needed
        "price_per_1k_input": 0.005,
        "price_per_1k_output": 0.015
    },
    "gpt-4-0125-preview": {
        "description": "standard",
        "max_tokens": 16384,  # Maximum API limit
        "safe_tokens": 14745,  # 90% of max for safety
        "max_safe_tokens": 16220,  # 99% of max for when needed
        "price_per_1k_input": 0.005,
        "price_per_1k_output": 0.015
    },
    "gpt-4-0613": {
        "description": "standard",
        "max_tokens": 16384,  # Maximum API limit
        "safe_tokens": 14745,  # 90% of max for safety
        "max_safe_tokens": 16220,  # 99% of max for when needed
        "price_per_1k_input": 0.005,
        "price_per_1k_output": 0.015
    },
    "gpt-4-1106-preview": {
        "description": "standard",
        "max_tokens": 16384,  # Maximum API limit
        "safe_tokens": 14745,  # 90% of max for safety
        "max_safe_tokens": 16220,  # 99% of max for when needed
        "price_per_1k_input": 0.005,
        "price_per_1k_output": 0.015
    },
    "gpt-4-turbo": {
        "description": "standard",
        "max_tokens": 16384,  # Maximum API limit
        "safe_tokens": 14745,  # 90% of max for safety
        "max_safe_tokens": 16220,  # 99% of max for when needed
        "price_per_1k_input": 0.005,
        "price_per_1k_output": 0.015
    },
    "gpt-4-turbo-2024-04-09": {
        "description": "standard",
        "max_tokens": 16384,  # Maximum API limit
        "safe_tokens": 14745,  # 90% of max for safety
        "max_safe_tokens": 16220,  # 99% of max for when needed
        "price_per_1k_input": 0.005,
        "price_per_1k_output": 0.015
    },
    "gpt-4-turbo-preview": {
        "description": "standard",
        "max_tokens": 16384,  # Maximum API limit
        "safe_tokens": 14745,  # 90% of max for safety
        "max_safe_tokens": 16220,  # 99% of max for when needed
        "price_per_1k_input": 0.005,
        "price_per_1k_output": 0.015
    },
    "gpt-4.1": {
        "description": "standard",
        "max_tokens": 16384,  # Maximum API limit
        "safe_tokens": 14745,  # 90% of max for safety
        "max_safe_tokens": 16220,  # 99% of max for when needed
        "price_per_1k_input": 0.005,
        "price_per_1k_output": 0.015
    },
    "gpt-4.1-2025-04-14": {
        "description": "standard",
        "max_tokens": 16384,  # Maximum API limit
        "safe_tokens": 14745,  # 90% of max for safety
        "max_safe_tokens": 16220,  # 99% of max for when needed
        "price_per_1k_input": 0.005,
        "price_per_1k_output": 0.015
    },
    "gpt-4.1-mini": {
        "description": "standard",
        "max_tokens": 16384,  # Maximum API limit
        "safe_tokens": 14745,  # 90% of max for safety
        "max_safe_tokens": 16220,  # 99% of max for when needed
        "price_per_1k_input": 0.005,
        "price_per_1k_output": 0.015
    },
    "gpt-4.1-mini-2025-04-14": {
        "description": "standard",
        "max_tokens": 16384,  # Maximum API limit
        "safe_tokens": 14745,  # 90% of max for safety
        "max_safe_tokens": 16220,  # 99% of max for when needed
        "price_per_1k_input": 0.005,
        "price_per_1k_output": 0.015
    },
    "gpt-4.1-nano": {
        "description": "standard",
        "max_tokens": 16384,  # Maximum API limit
        "safe_tokens": 14745,  # 90% of max for safety
        "max_safe_tokens": 16220,  # 99% of max for when needed
        "price_per_1k_input": 0.005,
        "price_per_1k_output": 0.015
    },
    "gpt-4.1-nano-2025-04-14": {
        "description": "standard",
        "max_tokens": 16384,  # Maximum API limit
        "safe_tokens": 14745,  # 90% of max for safety
        "max_safe_tokens": 16220,  # 99% of max for when needed
        "price_per_1k_input": 0.005,
        "price_per_1k_output": 0.015
    },
    "gpt-4.5-preview": {
        "description": "most powerful",
        "max_tokens": 16384,  # Maximum API limit
        "safe_tokens": 14745,  # 90% of max for safety
        "max_safe_tokens": 16220,  # 99% of max for when needed
        "price_per_1k_input": 0.01,
        "price_per_1k_output": 0.03
    },
    "gpt-4.5-preview-2025-02-27": {
        "description": "most powerful",
        "max_tokens": 16384,  # Maximum API limit
        "safe_tokens": 14745,  # 90% of max for safety
        "max_safe_tokens": 16220,  # 99% of max for when needed
        "price_per_1k_input": 0.01,
        "price_per_1k_output": 0.03
    },
    "gpt-4o": {
        "description": "powerful",
        "max_tokens": 16384,  # Maximum API limit
        "safe_tokens": 14745,  # 90% of max for safety
        "max_safe_tokens": 16220,  # 99% of max for when needed
        "price_per_1k_input": 0.005,
        "price_per_1k_output": 0.015
    },
    "gpt-4o-2024-05-13": {
        "description": "powerful",
        "max_tokens": 16384,  # Maximum API limit
        "safe_tokens": 14745,  # 90% of max for safety
        "max_safe_tokens": 16220,  # 99% of max for when needed
        "price_per_1k_input": 0.005,
        "price_per_1k_output": 0.015
    },
    "gpt-4o-2024-08-06": {
        "description": "powerful",
        "max_tokens": 16384,  # Maximum API limit
        "safe_tokens": 14745,  # 90% of max for safety
        "max_safe_tokens": 16220,  # 99% of max for when needed
        "price_per_1k_input": 0.005,
        "price_per_1k_output": 0.015
    },
    "gpt-4o-2024-11-20": {
        "description": "powerful",
        "max_tokens": 16384,  # Maximum API limit
        "safe_tokens": 14745,  # 90% of max for safety
        "max_safe_tokens": 16220,  # 99% of max for when needed
        "price_per_1k_input": 0.005,
        "price_per_1k_output": 0.015
    },
    "gpt-4o-audio-preview": {
        "description": "powerful",
        "max_tokens": 16384,  # Maximum API limit
        "safe_tokens": 14745,  # 90% of max for safety
        "max_safe_tokens": 16220,  # 99% of max for when needed
        "price_per_1k_input": 0.005,
        "price_per_1k_output": 0.015
    },
    "gpt-4o-audio-preview-2024-10-01": {
        "description": "powerful",
        "max_tokens": 16384,  # Maximum API limit
        "safe_tokens": 14745,  # 90% of max for safety
        "max_safe_tokens": 16220,  # 99% of max for when needed
        "price_per_1k_input": 0.005,
        "price_per_1k_output": 0.015
    },
    "gpt-4o-audio-preview-2024-12-17": {
        "description": "powerful",
        "max_tokens": 16384,  # Maximum API limit
        "safe_tokens": 14745,  # 90% of max for safety
        "max_safe_tokens": 16220,  # 99% of max for when needed
        "price_per_1k_input": 0.005,
        "price_per_1k_output": 0.015
    },
    "gpt-4o-mini": {
        "description": "balanced",
        "max_tokens": 16384,  # Maximum API limit
        "safe_tokens": 14745,  # 90% of max for safety
        "max_safe_tokens": 16220,  # 99% of max for when needed
        "price_per_1k_input": 0.0015,
        "price_per_1k_output": 0.006
    },
    "gpt-4o-mini-2024-07-18": {
        "description": "balanced",
        "max_tokens": 16384,  # Maximum API limit
        "safe_tokens": 14745,  # 90% of max for safety
        "max_safe_tokens": 16220,  # 99% of max for when needed
        "price_per_1k_input": 0.005,
        "price_per_1k_output": 0.015
    },
    "gpt-4o-mini-audio-preview": {
        "description": "balanced",
        "max_tokens": 16384,  # Maximum API limit
        "safe_tokens": 14745,  # 90% of max for safety
        "max_safe_tokens": 16220,  # 99% of max for when needed
        "price_per_1k_input": 0.005,
        "price_per_1k_output": 0.015
    },
    "gpt-4o-mini-audio-preview-2024-12-17": {
        "description": "balanced",
        "max_tokens": 16384,  # Maximum API limit
        "safe_tokens": 14745,  # 90% of max for safety
        "max_safe_tokens": 16220,  # 99% of max for when needed
        "price_per_1k_input": 0.005,
        "price_per_1k_output": 0.015
    },
    "gpt-4o-mini-realtime-preview": {
        "description": "balanced",
        "max_tokens": 16384,  # Maximum API limit
        "safe_tokens": 14745,  # 90% of max for safety
        "max_safe_tokens": 16220,  # 99% of max for when needed
        "price_per_1k_input": 0.005,
        "price_per_1k_output": 0.015
    },
    "gpt-4o-mini-realtime-preview-2024-12-17": {
        "description": "balanced",
        "max_tokens": 16384,  # Maximum API limit
        "safe_tokens": 14745,  # 90% of max for safety
        "max_safe_tokens": 16220,  # 99% of max for when needed
        "price_per_1k_input": 0.005,
        "price_per_1k_output": 0.015
    },
    "gpt-4o-mini-search-preview": {
        "description": "balanced",
        "max_tokens": 16384,  # Maximum API limit
        "safe_tokens": 14745,  # 90% of max for safety
        "max_safe_tokens": 16220,  # 99% of max for when needed
        "price_per_1k_input": 0.005,
        "price_per_1k_output": 0.015
    },
    "gpt-4o-mini-search-preview-2025-03-11": {
        "description": "balanced",
        "max_tokens": 16384,  # Maximum API limit
        "safe_tokens": 14745,  # 90% of max for safety
        "max_safe_tokens": 16220,  # 99% of max for when needed
        "price_per_1k_input": 0.005,
        "price_per_1k_output": 0.015
    },
    "gpt-4o-mini-transcribe": {
        "description": "balanced",
        "max_tokens": 16384,  # Maximum API limit
        "safe_tokens": 14745,  # 90% of max for safety
        "max_safe_tokens": 16220,  # 99% of max for when needed
        "price_per_1k_input": 0.005,
        "price_per_1k_output": 0.015
    },
    "gpt-4o-mini-tts": {
        "description": "balanced",
        "max_tokens": 16384,  # Maximum API limit
        "safe_tokens": 14745,  # 90% of max for safety
        "max_safe_tokens": 16220,  # 99% of max for when needed
        "price_per_1k_input": 0.005,
        "price_per_1k_output": 0.015
    },
    "gpt-4o-realtime-preview": {
        "description": "powerful",
        "max_tokens": 16384,  # Maximum API limit
        "safe_tokens": 14745,  # 90% of max for safety
        "max_safe_tokens": 16220,  # 99% of max for when needed
        "price_per_1k_input": 0.005,
        "price_per_1k_output": 0.015
    },
    "gpt-4o-realtime-preview-2024-10-01": {
        "description": "powerful",
        "max_tokens": 16384,  # Maximum API limit
        "safe_tokens": 14745,  # 90% of max for safety
        "max_safe_tokens": 16220,  # 99% of max for when needed
        "price_per_1k_input": 0.005,
        "price_per_1k_output": 0.015
    },
    "gpt-4o-realtime-preview-2024-12-17": {
        "description": "powerful",
        "max_tokens": 16384,  # Maximum API limit
        "safe_tokens": 14745,  # 90% of max for safety
        "max_safe_tokens": 16220,  # 99% of max for when needed
        "price_per_1k_input": 0.005,
        "price_per_1k_output": 0.015
    },
    "gpt-4o-search-preview": {
        "description": "powerful",
        "max_tokens": 16384,  # Maximum API limit
        "safe_tokens": 14745,  # 90% of max for safety
        "max_safe_tokens": 16220,  # 99% of max for when needed
        "price_per_1k_input": 0.005,
        "price_per_1k_output": 0.015
    },
    "gpt-4o-search-preview-2025-03-11": {
        "description": "powerful",
        "max_tokens": 16384,  # Maximum API limit
        "safe_tokens": 14745,  # 90% of max for safety
        "max_safe_tokens": 16220,  # 99% of max for when needed
        "price_per_1k_input": 0.005,
        "price_per_1k_output": 0.015
    },
    "gpt-4o-transcribe": {
        "description": "powerful",
        "max_tokens": 16384,  # Maximum API limit
        "safe_tokens": 14745,  # 90% of max for safety
        "max_safe_tokens": 16220,  # 99% of max for when needed
        "price_per_1k_input": 0.005,
        "price_per_1k_output": 0.015
    },
    "o1": {
        "description": "powerful reasoning",
        "max_tokens": 16384,  # Maximum API limit
        "safe_tokens": 14745,  # 90% of max for safety
        "max_safe_tokens": 16220,  # 99% of max for when needed
        "price_per_1k_input": 0.015,
        "price_per_1k_output": 0.075
    },
    "o1-mini": {
        "description": "standard",
        "max_tokens": 16384,  # Maximum API limit
        "safe_tokens": 14745,  # 90% of max for safety
        "max_safe_tokens": 16220,  # 99% of max for when needed
        "price_per_1k_input": 0.005,
        "price_per_1k_output": 0.015
    },
    "o1-pro": {
        "description": "standard",
        "max_tokens": 16384,  # Maximum API limit
        "safe_tokens": 14745,  # 90% of max for safety
        "max_safe_tokens": 16220,  # 99% of max for when needed
        "price_per_1k_input": 0.005,
        "price_per_1k_output": 0.015
    },
    "o3": {
        "description": "powerful reasoning",
        "max_tokens": 16384,  # Maximum API limit
        "safe_tokens": 14745,  # 90% of max for safety
        "max_safe_tokens": 16220,  # 99% of max for when needed
        "price_per_1k_input": 0.015,
        "price_per_1k_output": 0.075
    },
    "o3-mini": {
        "description": "standard",
        "max_tokens": 16384,  # Maximum API limit
        "safe_tokens": 14745,  # 90% of max for safety
        "max_safe_tokens": 16220,  # 99% of max for when needed
        "price_per_1k_input": 0.005,
        "price_per_1k_output": 0.015
    },
    "o4-mini": {
        "description": "fast reasoning",
        "max_tokens": 16384,  # Maximum API limit
        "safe_tokens": 14745,  # 90% of max for safety
        "max_safe_tokens": 16220,  # 99% of max for when needed
        "price_per_1k_input": 0.005,
        "price_per_1k_output": 0.015
    },
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

def get_max_tokens_for_model(model_name, use_safe_limit=True, use_max_safe=False):
    """Get token limit for a model based on safety preferences.
    
    Args:
        model_name: Name of the model
        use_safe_limit: Whether to use the 90% safe limit (default: True)
        use_max_safe: Whether to use the 99% max safe limit (default: False)
        
    Returns:
        The appropriate token limit based on safety preferences
    """
    model_info = get_openai_model_info(model_name)
    if not model_info:
        return 4096  # Default fallback
        
    if use_max_safe:
        return model_info.get("max_safe_tokens", model_info.get("max_tokens", 4096))
    elif use_safe_limit:
        return model_info.get("safe_tokens", model_info.get("max_tokens", 4096))
    else:
        return model_info.get("max_tokens", 4096)

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
