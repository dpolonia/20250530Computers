"""Client classes for interacting with various LLM APIs."""

import os
import time
import anthropic
import openai
import google.generativeai as genai
import logging
from typing import Dict, List, Optional, Any, Union

from src.security.credential_manager import get_credential_manager, CredentialManager
from src.security.input_validator import (
    validate_string_input, 
    validate_api_key, 
    ValidationError
)

# Configure logging
logger = logging.getLogger(__name__)

class BaseLLMClient:
    """Base class for LLM API clients."""
    
    def __init__(self):
        """Initialize the base client."""
        self.provider = "base"
        self.model = None
        self.total_tokens_used = 0
        self.total_cost = 0.0
        self.request_count = 0
        self.verified = False
        self.credential_manager = get_credential_manager()
    
    def validate_api_key(self) -> bool:
        """Validate the API key.
        
        Returns:
            True if the API key is valid, False otherwise
        """
        raise NotImplementedError("Subclasses must implement validate_api_key")
    
    def verify_model_accuracy(self) -> bool:
        """Verify model accuracy with a simple factual question.
        
        Tests if the model correctly answers "In what day is Christmas?"
        
        Returns:
            True if the model correctly answers December 25th, False otherwise
        """
        try:
            # Test prompt for model verification
            test_prompt = "In what day is Christmas?"
            system_prompt = "You are a helpful assistant. Answer the question concisely in one sentence."
            
            # Get response from model - IMPORTANT: Skip verification to avoid infinite recursion
            # This is a direct call that bypasses the verification check in get_completion
            response = self._get_completion_without_verification(
                prompt=test_prompt, 
                system_prompt=system_prompt,
                max_tokens=100
            )
            
            # Check if response contains "December 25" or "25th of December" or similar
            response_lower = response.lower()
            correct_answer = any(phrase in response_lower for phrase in [
                "december 25", "25th of december", "25 december", 
                "december 25th", "25 dec", "dec 25", "25th dec"
            ])
            
            # Print verification result
            if correct_answer:
                logger.info(f"{self.provider.upper()} {self.model} VERIFICATION PASSED")
                logger.debug(f"Question: {test_prompt}")
                logger.debug(f"Response: {response}")
                self.verified = True
            else:
                logger.warning(f"{self.provider.upper()} {self.model} VERIFICATION FAILED")
                logger.warning(f"Question: {test_prompt}")
                logger.warning(f"Response: {response}")
                logger.warning(f"ALERT: Model did not answer 'December 25th' or similar")
                self.verified = False
            
            return correct_answer
        except Exception as e:
            logger.error(f"{self.provider.upper()} {self.model} VERIFICATION ERROR: {str(e)}")
            self.verified = False
            return False
    
    def _get_completion_without_verification(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """Get a completion from the model without verification.
        
        This method is used internally by the verification system to avoid infinite recursion.
        Subclasses must implement this method to provide the actual completion logic.
        
        Args:
            prompt: The user prompt to send to the model
            system_prompt: Optional system prompt
            **kwargs: Additional parameters for the model
            
        Returns:
            The model's response as a string
        """
        raise NotImplementedError("Subclasses must implement _get_completion_without_verification")
    
    def get_completion(self, prompt: str, system_prompt: Optional[str] = None, 
                     verify_first: bool = True, **kwargs) -> str:
        """Get a completion from the model.
        
        Args:
            prompt: The user prompt to send to the model
            system_prompt: Optional system prompt
            verify_first: Whether to verify model accuracy before proceeding (default: True)
            **kwargs: Additional parameters for the model
            
        Returns:
            The model's response as a string
            
        Raises:
            RuntimeError: If model verification fails and user chooses not to continue
        """
        # Validate inputs
        try:
            prompt = validate_string_input(prompt, min_length=1)
            if system_prompt:
                system_prompt = validate_string_input(system_prompt, min_length=1)
        except ValidationError as e:
            logger.error(f"Input validation error: {e}")
            raise
            
        # Verify model accuracy before proceeding if requested
        if verify_first and not self.verified:
            logger.info(f"Verifying {self.provider.upper()} model {self.model} before using...")
            if not self.verify_model_accuracy():
                # Model failed verification
                choice = input("Model verification failed. Continue anyway? (y/n): ")
                if choice.lower() != 'y':
                    raise RuntimeError(f"{self.provider.upper()} model {self.model} failed verification check")
        
        # Use the implementation in subclasses
        return self._get_completion_without_verification(prompt, system_prompt, **kwargs)
    
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
        stats = {
            "provider": self.provider,
            "model": self.model,
            "total_tokens": self.total_tokens_used,
            "total_cost": self.total_cost,
            "request_count": self.request_count
        }
        
        # Format a human-readable summary for logging
        stats["summary"] = (
            f"[{self.provider.upper()}] Model: {self.model}\n"
            f"Total requests: {self.request_count}\n"
            f"Total tokens used: {self.total_tokens_used:,}\n"
            f"Total cost: ${self.total_cost:.6f}"
        )
        
        return stats

class AnthropicClient(BaseLLMClient):
    """Client for interacting with Anthropic Claude API."""
    
    def __init__(self, model_name: str):
        """Initialize the Anthropic client.
        
        Args:
            model_name: Name of the Claude model to use
        """
        super().__init__()
        self.provider = "anthropic"
        
        # Safely extract model name
        try:
            self.model = model_name.split(" (")[0].strip()  # Extract model name without description
        except (AttributeError, IndexError):
            self.model = str(model_name).strip()
            
        # Get API key from credential manager
        self.api_key = self.credential_manager.get_credential(
            CredentialManager.ANTHROPIC_API_KEY,
            env_var="ANTHROPIC_API_KEY"
        )
        
        # Validate API key
        try:
            self.api_key = validate_api_key(self.api_key, min_length=8)
        except ValidationError as e:
            logger.error(f"API key validation error: {e}")
            raise ValueError(f"Invalid Anthropic API key: {e}")
            
        # Initialize client with validated API key
        self.client = anthropic.Anthropic(api_key=self.api_key)
        
        # Price per 1K tokens (simplified - actual pricing may vary)
        from src.models.anthropic_models import get_claude_model_info
        self.model_info = get_claude_model_info(model_name)
        self.price_per_1k_input = self.model_info.get("price_per_1k_input", 0.0025)
        self.price_per_1k_output = self.model_info.get("price_per_1k_output", 0.0125)
    
    def validate_api_key(self) -> bool:
        """Validate the Anthropic API key."""
        if not self.api_key:
            logger.warning("Anthropic API key is missing")
            return False
            
        try:
            # Make a minimal API call to validate the key
            self.client.messages.create(
                model=self.model,
                max_tokens=1,
                messages=[{"role": "user", "content": "Hello"}]
            )
            logger.info("Anthropic API key is valid")
            return True
        except Exception as e:
            logger.error(f"Anthropic API key validation failed: {e}")
            return False
    
    def _get_completion_without_verification(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """Get a completion from Claude without verification.
        
        This implementation contains the actual logic to call the model.
        
        Args:
            prompt: The user prompt to send to the model
            system_prompt: Optional system prompt
            **kwargs: Additional parameters for the model
            
        Returns:
            The model's response as a string
        """
        system = system_prompt or "You are a helpful AI assistant."
        
        max_tokens = kwargs.get("max_tokens", 4096)
        temperature = kwargs.get("temperature", 0.7)
        
        start_time = time.time()
        
        # Use streaming for longer operations to avoid timeouts
        use_streaming = max_tokens > 1000 or len(prompt) > 4000
        
        try:
            if use_streaming:
                # Streaming approach
                full_response = ""
                logger.info("Streaming response (this may take a moment)...")
                
                with self.client.messages.stream(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system,
                    messages=[{"role": "user", "content": prompt}]
                ) as stream:
                    # Track progress with dots
                    progress_counter = 0
                    
                    for chunk in stream:
                        if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'text') and chunk.delta.text:
                            full_response += chunk.delta.text
                            progress_counter += len(chunk.delta.text)
                        elif hasattr(chunk, 'content_block') and hasattr(chunk.content_block, 'text'):
                            full_response += chunk.content_block.text
                            progress_counter += len(chunk.content_block.text)
                        
                        # Log progress periodically
                        if progress_counter >= 200:
                            logger.debug("Streaming in progress...")
                            progress_counter = 0
                    
                    # Get the final message for token counting
                    response = stream.get_final_message()
                    logger.info("Streaming complete")
            else:
                # Non-streaming approach for shorter requests
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system,
                    messages=[{"role": "user", "content": prompt}]
                )
                full_response = response.content[0].text
                
            elapsed_time = time.time() - start_time
            
            # Update stats
            token_usage = self.get_tokens_from_response(response)
            self.total_tokens_used += token_usage["total_tokens"]
            input_cost = (token_usage["input_tokens"] / 1000) * self.price_per_1k_input
            output_cost = (token_usage["output_tokens"] / 1000) * self.price_per_1k_output
            self.total_cost += input_cost + output_cost
            self.request_count += 1
            
            # Log usage
            logger.info(
                f"[{self.provider.upper()}] Request completed in {elapsed_time:.2f}s. "
                f"Used {token_usage['total_tokens']} tokens "
                f"(Input: {token_usage['input_tokens']}, Output: {token_usage['output_tokens']}), "
                f"cost: ${input_cost + output_cost:.6f}"
            )
            
            return full_response
            
        except Exception as e:
            logger.error(f"Error getting completion from Anthropic: {e}")
            raise
        
    def get_completion(self, prompt: str, system_prompt: Optional[str] = None, 
                     verify_first: bool = True, **kwargs) -> str:
        """Get a completion from Claude.
        
        Args:
            prompt: The user prompt to send to the model
            system_prompt: Optional system prompt
            verify_first: Whether to verify model accuracy before proceeding (default: True)
            **kwargs: Additional parameters for the model
            
        Returns:
            The model's response as a string
            
        Raises:
            RuntimeError: If model verification fails and user chooses not to continue
        """
        # Use the parent class method which handles verification
        return super().get_completion(prompt, system_prompt, verify_first, **kwargs)
    
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
        
        # Safely extract model name
        try:
            self.model = model_name.split(" (")[0].strip()  # Extract model name without description
        except (AttributeError, IndexError):
            self.model = str(model_name).strip()
            
        # Get API key from credential manager
        self.api_key = self.credential_manager.get_credential(
            CredentialManager.OPENAI_API_KEY,
            env_var="OPENAI_API_KEY"
        )
        
        # Validate API key
        try:
            self.api_key = validate_api_key(self.api_key, min_length=8)
        except ValidationError as e:
            logger.error(f"API key validation error: {e}")
            raise ValueError(f"Invalid OpenAI API key: {e}")
            
        # Initialize client with validated API key
        self.client = openai.OpenAI(api_key=self.api_key)
        
        # Price per 1K tokens (simplified - actual pricing may vary)
        from src.models.openai_models import get_openai_model_info
        self.model_info = get_openai_model_info(model_name)
        self.price_per_1k_input = self.model_info.get("price_per_1k_input", 0.001)
        self.price_per_1k_output = self.model_info.get("price_per_1k_output", 0.002)
    
    def validate_api_key(self) -> bool:
        """Validate the OpenAI API key."""
        if not self.api_key:
            logger.warning("OpenAI API key is missing")
            return False
            
        try:
            # Make a minimal API call to validate the key
            self.client.chat.completions.create(
                model=self.model,
                max_tokens=1,
                messages=[{"role": "user", "content": "Hello"}]
            )
            logger.info("OpenAI API key is valid")
            return True
        except Exception as e:
            logger.error(f"OpenAI API key validation failed: {e}")
            return False
    
    def _get_completion_without_verification(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """Get a completion from OpenAI without verification.
        
        This implementation contains the actual logic to call the model.
        
        Args:
            prompt: The user prompt to send to the model
            system_prompt: Optional system prompt
            **kwargs: Additional parameters for the model
            
        Returns:
            The model's response as a string
        """
        system = system_prompt or "You are a helpful AI assistant."
        
        max_tokens = kwargs.get("max_tokens", 4096)
        temperature = kwargs.get("temperature", 0.7)
        
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ]
        
        try:
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
            logger.info(
                f"[{self.provider.upper()}] Request completed in {elapsed_time:.2f}s. "
                f"Used {token_usage['total_tokens']} tokens "
                f"(Input: {token_usage['input_tokens']}, Output: {token_usage['output_tokens']}), "
                f"cost: ${input_cost + output_cost:.6f}"
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error getting completion from OpenAI: {e}")
            raise
    
    def get_completion(self, prompt: str, system_prompt: Optional[str] = None, 
                     verify_first: bool = True, **kwargs) -> str:
        """Get a completion from OpenAI.
        
        Args:
            prompt: The user prompt to send to the model
            system_prompt: Optional system prompt
            verify_first: Whether to verify model accuracy before proceeding (default: True)
            **kwargs: Additional parameters for the model
            
        Returns:
            The model's response as a string
            
        Raises:
            RuntimeError: If model verification fails and user chooses not to continue
        """
        # Use the parent class method which handles verification
        return super().get_completion(prompt, system_prompt, verify_first, **kwargs)
    
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
        
        # Safely extract model name
        try:
            self.model = model_name.split(" (")[0].strip()  # Extract model name without description
        except (AttributeError, IndexError):
            self.model = str(model_name).strip()
            
        # Get API key from credential manager
        self.api_key = self.credential_manager.get_credential(
            CredentialManager.GOOGLE_API_KEY,
            env_var="GOOGLE_API_KEY"
        )
        
        # Validate API key
        try:
            self.api_key = validate_api_key(self.api_key, min_length=8)
        except ValidationError as e:
            logger.error(f"API key validation error: {e}")
            raise ValueError(f"Invalid Google API key: {e}")
            
        # Initialize Gemini
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
            logger.warning("Google API key is missing")
            return False
            
        try:
            # Make a minimal API call to validate the key
            models = genai.list_models()
            model_count = len(list(models))
            logger.info(f"Google API key is valid. Found {model_count} models.")
            return model_count > 0
        except Exception as e:
            logger.error(f"Google API key validation failed: {e}")
            return False
    
    def _get_completion_without_verification(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """Get a completion from Gemini without verification.
        
        This implementation contains the actual logic to call the model.
        
        Args:
            prompt: The user prompt to send to the model
            system_prompt: Optional system prompt
            **kwargs: Additional parameters for the model
            
        Returns:
            The model's response as a string
        """
        system = system_prompt or "You are a helpful AI assistant."
        
        # Update generation config with kwargs
        generation_config = self.generation_config.copy()
        if "max_tokens" in kwargs:
            generation_config["max_output_tokens"] = kwargs["max_tokens"]
        if "temperature" in kwargs:
            generation_config["temperature"] = kwargs["temperature"]
        
        # Combine system prompt and user prompt
        full_prompt = f"{system}\n\n{prompt}"
        
        # Store the prompt for token estimation
        self._last_prompt = full_prompt
        
        try:
            start_time = time.time()
            response = self.client.generate_content(
                full_prompt,
                generation_config=generation_config
            )
            elapsed_time = time.time() - start_time
            
            # Update stats using more accurate token estimation
            token_usage = self.get_tokens_from_response(response)
            self.total_tokens_used += token_usage["total_tokens"]
            input_cost = (token_usage["input_tokens"] / 1000) * self.price_per_1k_input
            output_cost = (token_usage["output_tokens"] / 1000) * self.price_per_1k_output
            self.total_cost += input_cost + output_cost
            self.request_count += 1
            
            # Log usage
            logger.info(
                f"[{self.provider.upper()}] Request completed in {elapsed_time:.2f}s. "
                f"Used approximately {token_usage['total_tokens']} tokens "
                f"(Input: {token_usage['input_tokens']}, Output: {token_usage['output_tokens']}), "
                f"cost: ${input_cost + output_cost:.6f}"
            )
            
            return response.text
            
        except Exception as e:
            logger.error(f"Error getting completion from Gemini: {e}")
            raise
    
    def get_completion(self, prompt: str, system_prompt: Optional[str] = None, 
                     verify_first: bool = True, **kwargs) -> str:
        """Get a completion from Gemini.
        
        Args:
            prompt: The user prompt to send to the model
            system_prompt: Optional system prompt
            verify_first: Whether to verify model accuracy before proceeding (default: True)
            **kwargs: Additional parameters for the model
            
        Returns:
            The model's response as a string
            
        Raises:
            RuntimeError: If model verification fails and user chooses not to continue
        """
        # Use the parent class method which handles verification
        return super().get_completion(prompt, system_prompt, verify_first, **kwargs)
    
    def get_tokens_from_response(self, response: Any) -> Dict[str, int]:
        """Estimate token usage for Gemini (not directly provided by API).
        
        Gemini API doesn't provide token counts directly in the response,
        but we can use a more accurate estimation based on character count.
        """
        # Approximate token counts based on model response
        # Gemini uses ~4 characters per token on average
        if hasattr(response, 'candidates') and response.candidates:
            # For structured response
            text = response.candidates[0].content.parts[0].text
        else:
            # For simple response
            text = response.text
        
        # Input tokens estimation from the stored prompt
        prompt_chars = len(getattr(self, '_last_prompt', ''))
        input_tokens = prompt_chars // 4  # ~4 chars per token
        
        # Output tokens from response text
        output_chars = len(text)
        output_tokens = output_chars // 4  # ~4 chars per token
        
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens
        }

def get_llm_client(provider: str, model_name: str, verify: bool = True) -> BaseLLMClient:
    """Factory function to get the appropriate LLM client.
    
    Args:
        provider: Provider name ("anthropic", "openai", or "google")
        model_name: Name of the model to use
        verify: Whether to verify model accuracy on initialization (default: True)
        
    Returns:
        An LLM client instance
        
    Raises:
        ValueError: If the provider is not supported
    """
    # Validate inputs
    try:
        provider = validate_string_input(
            provider, 
            allowed_chars="abcdefghijklmnopqrstuvwxyz",
            case_sensitive=False
        ).lower()
        
        model_name = validate_string_input(model_name, min_length=1)
    except ValidationError as e:
        logger.error(f"Input validation error: {e}")
        raise ValueError(f"Invalid input: {e}")
    
    # Validate provider choice
    valid_providers = ["anthropic", "openai", "google"]
    if provider not in valid_providers:
        raise ValueError(
            f"Unsupported provider: {provider}. "
            f"Supported providers: {', '.join(valid_providers)}"
        )
    
    # Create appropriate client
    client = None
    
    try:
        if provider == "anthropic":
            client = AnthropicClient(model_name)
        elif provider == "openai":
            client = OpenAIClient(model_name)
        elif provider == "google":
            client = GeminiClient(model_name)
        
        # Verify model accuracy if requested
        if verify:
            logger.info(f"Verifying {provider.upper()} model {model_name}...")
            client.verify_model_accuracy()
        
        return client
        
    except Exception as e:
        logger.error(f"Error creating LLM client for {provider}: {e}")
        raise