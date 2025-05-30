"""Client classes for interacting with various LLM APIs."""

import os
import time
import anthropic
import openai
import google.generativeai as genai
from typing import Dict, List, Optional, Any

class BaseLLMClient:
    """Base class for LLM API clients."""
    
    def __init__(self):
        """Initialize the base client."""
        self.provider = "base"
        self.model = None
        self.total_tokens_used = 0
        self.total_cost = 0.0
        self.request_count = 0
    
    def validate_api_key(self) -> bool:
        """Validate the API key.
        
        Returns:
            True if the API key is valid, False otherwise
        """
        raise NotImplementedError("Subclasses must implement validate_api_key")
    
    def get_completion(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """Get a completion from the model.
        
        Args:
            prompt: The user prompt to send to the model
            system_prompt: Optional system prompt
            **kwargs: Additional parameters for the model
            
        Returns:
            The model's response as a string
        """
        raise NotImplementedError("Subclasses must implement get_completion")
    
    def get_tokens_from_response(self, response: Any) -> Dict[str, int]:
        """Extract token usage from a response.
        
        Args:
            response: The raw response from the model
            
        Returns:
            Dictionary with token usage information
        """
        raise NotImplementedError("Subclasses must implement get_tokens_from_response")
    
    def get_usage_statistics(self) -> Dict[str, Any]:
        """Get usage statistics.
        
        Returns:
            Dictionary with usage statistics
        """
        return {
            "provider": self.provider,
            "model": self.model,
            "total_tokens": self.total_tokens_used,
            "total_cost": self.total_cost,
            "request_count": self.request_count
        }

class AnthropicClient(BaseLLMClient):
    """Client for interacting with Anthropic Claude API."""
    
    def __init__(self, model_name: str):
        """Initialize the Anthropic client.
        
        Args:
            model_name: Name of the Claude model to use
        """
        super().__init__()
        self.provider = "anthropic"
        self.model = model_name.split(" (")[0]  # Extract model name without description
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=self.api_key)
        
        # Price per 1K tokens (simplified - actual pricing may vary)
        from src.models.anthropic_models import get_claude_model_info
        self.model_info = get_claude_model_info(model_name)
        self.price_per_1k_input = self.model_info.get("price_per_1k_input", 0.0025)
        self.price_per_1k_output = self.model_info.get("price_per_1k_output", 0.0125)
    
    def validate_api_key(self) -> bool:
        """Validate the Anthropic API key."""
        if not self.api_key:
            return False
            
        try:
            self.client.messages.create(
                model=self.model,
                max_tokens=1,
                messages=[{"role": "user", "content": "Hello"}]
            )
            return True
        except Exception:
            return False
    
    def get_completion(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """Get a completion from Claude."""
        system = system_prompt or "You are a helpful AI assistant."
        
        max_tokens = kwargs.get("max_tokens", 4096)
        temperature = kwargs.get("temperature", 0.7)
        
        start_time = time.time()
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": prompt}]
        )
        elapsed_time = time.time() - start_time
        
        # Update stats
        token_usage = self.get_tokens_from_response(response)
        self.total_tokens_used += token_usage["total_tokens"]
        input_cost = (token_usage["input_tokens"] / 1000) * self.price_per_1k_input
        output_cost = (token_usage["output_tokens"] / 1000) * self.price_per_1k_output
        self.total_cost += input_cost + output_cost
        self.request_count += 1
        
        # Log usage
        print(f"Request completed in {elapsed_time:.2f}s. Used {token_usage['total_tokens']} tokens, cost: ${input_cost + output_cost:.6f}")
        
        return response.content[0].text
    
    def get_tokens_from_response(self, response: Any) -> Dict[str, int]:
        """Extract token usage from an Anthropic response."""
        return {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens
        }

class OpenAIClient(BaseLLMClient):
    """Client for interacting with OpenAI API."""
    
    def __init__(self, model_name: str):
        """Initialize the OpenAI client.
        
        Args:
            model_name: Name of the OpenAI model to use
        """
        super().__init__()
        self.provider = "openai"
        self.model = model_name.split(" (")[0]  # Extract model name without description
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = openai.OpenAI(api_key=self.api_key)
        
        # Price per 1K tokens (simplified - actual pricing may vary)
        from src.models.openai_models import get_openai_model_info
        self.model_info = get_openai_model_info(model_name)
        self.price_per_1k_input = self.model_info.get("price_per_1k_input", 0.001)
        self.price_per_1k_output = self.model_info.get("price_per_1k_output", 0.002)
    
    def validate_api_key(self) -> bool:
        """Validate the OpenAI API key."""
        if not self.api_key:
            return False
            
        try:
            self.client.chat.completions.create(
                model=self.model,
                max_tokens=1,
                messages=[{"role": "user", "content": "Hello"}]
            )
            return True
        except Exception:
            return False
    
    def get_completion(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """Get a completion from OpenAI."""
        system = system_prompt or "You are a helpful AI assistant."
        
        max_tokens = kwargs.get("max_tokens", 4096)
        temperature = kwargs.get("temperature", 0.7)
        
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ]
        
        start_time = time.time()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        elapsed_time = time.time() - start_time
        
        # Update stats
        token_usage = self.get_tokens_from_response(response)
        self.total_tokens_used += token_usage["total_tokens"]
        input_cost = (token_usage["input_tokens"] / 1000) * self.price_per_1k_input
        output_cost = (token_usage["output_tokens"] / 1000) * self.price_per_1k_output
        self.total_cost += input_cost + output_cost
        self.request_count += 1
        
        # Log usage
        print(f"Request completed in {elapsed_time:.2f}s. Used {token_usage['total_tokens']} tokens, cost: ${input_cost + output_cost:.6f}")
        
        return response.choices[0].message.content
    
    def get_tokens_from_response(self, response: Any) -> Dict[str, int]:
        """Extract token usage from an OpenAI response."""
        return {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }

class GeminiClient(BaseLLMClient):
    """Client for interacting with Google Gemini API."""
    
    def __init__(self, model_name: str):
        """Initialize the Gemini client.
        
        Args:
            model_name: Name of the Gemini model to use
        """
        super().__init__()
        self.provider = "google"
        self.model = model_name.split(" (")[0]  # Extract model name without description
        self.api_key = os.getenv("GOOGLE_API_KEY")
        genai.configure(api_key=self.api_key)
        
        # Price per 1K tokens (simplified - actual pricing may vary)
        from src.models.google_models import get_gemini_model_info
        self.model_info = get_gemini_model_info(model_name)
        self.price_per_1k_input = self.model_info.get("price_per_1k_input", 0.0005)
        self.price_per_1k_output = self.model_info.get("price_per_1k_output", 0.0005)
        
        self.generation_config = {
            "temperature": 0.7,
            "top_p": 1.0,
            "top_k": 1,
            "max_output_tokens": 4096,
        }
        
        # Initialize the model
        self.client = genai.GenerativeModel(self.model)
    
    def validate_api_key(self) -> bool:
        """Validate the Google API key."""
        if not self.api_key:
            return False
            
        try:
            models = genai.list_models()
            return len(list(models)) > 0
        except Exception:
            return False
    
    def get_completion(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """Get a completion from Gemini."""
        system = system_prompt or "You are a helpful AI assistant."
        
        # Update generation config with kwargs
        generation_config = self.generation_config.copy()
        if "max_tokens" in kwargs:
            generation_config["max_output_tokens"] = kwargs["max_tokens"]
        if "temperature" in kwargs:
            generation_config["temperature"] = kwargs["temperature"]
        
        # Combine system prompt and user prompt
        full_prompt = f"{system}\n\n{prompt}"
        
        start_time = time.time()
        response = self.client.generate_content(
            full_prompt,
            generation_config=generation_config
        )
        elapsed_time = time.time() - start_time
        
        # Update stats
        # Gemini API doesn't provide token counts directly, so we estimate
        estimated_input_tokens = len(full_prompt.split()) * 1.5  # rough estimate
        estimated_output_tokens = len(response.text.split()) * 1.5  # rough estimate
        estimated_total_tokens = estimated_input_tokens + estimated_output_tokens
        
        self.total_tokens_used += estimated_total_tokens
        input_cost = (estimated_input_tokens / 1000) * self.price_per_1k_input
        output_cost = (estimated_output_tokens / 1000) * self.price_per_1k_output
        self.total_cost += input_cost + output_cost
        self.request_count += 1
        
        # Log usage
        print(f"Request completed in {elapsed_time:.2f}s. Estimated {estimated_total_tokens} tokens, cost: ${input_cost + output_cost:.6f}")
        
        return response.text
    
    def get_tokens_from_response(self, response: Any) -> Dict[str, int]:
        """Estimate token usage for Gemini (not directly provided by API)."""
        # Rough estimation as Gemini doesn't provide token counts
        estimated_input_tokens = 0
        estimated_output_tokens = len(response.text.split()) * 1.5  # rough estimate
        
        return {
            "input_tokens": estimated_input_tokens,
            "output_tokens": estimated_output_tokens,
            "total_tokens": estimated_input_tokens + estimated_output_tokens
        }

def get_llm_client(provider: str, model_name: str) -> BaseLLMClient:
    """Factory function to get the appropriate LLM client.
    
    Args:
        provider: Provider name ("anthropic", "openai", or "google")
        model_name: Name of the model to use
        
    Returns:
        An LLM client instance
        
    Raises:
        ValueError: If the provider is not supported
    """
    if provider == "anthropic":
        return AnthropicClient(model_name)
    elif provider == "openai":
        return OpenAIClient(model_name)
    elif provider == "google":
        return GeminiClient(model_name)
    else:
        raise ValueError(f"Unsupported provider: {provider}")
