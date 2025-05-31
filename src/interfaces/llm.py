"""
Language Model interfaces for the Paper Revision System.

This module defines interfaces for language model components,
including clients for different providers and service abstractions.
"""

import abc
from typing import Dict, List, Optional, Any, Union


class LLMClientInterface(abc.ABC):
    """Interface for language model clients."""
    
    @abc.abstractmethod
    def validate_api_key(self) -> bool:
        """Validate the API key.
        
        Returns:
            True if the API key is valid, False otherwise
        """
        pass
    
    @abc.abstractmethod
    def get_completion(self, prompt: str, system_prompt: Optional[str] = None, 
                     verify_first: bool = True, **kwargs) -> str:
        """Get a completion from the model.
        
        Args:
            prompt: The user prompt to send to the model
            system_prompt: Optional system prompt
            verify_first: Whether to verify model accuracy before proceeding
            **kwargs: Additional parameters for the model
            
        Returns:
            The model's response as a string
        """
        pass
    
    @abc.abstractmethod
    def get_tokens_from_response(self, response: Any) -> Dict[str, int]:
        """Extract token usage from a response.
        
        Args:
            response: The raw response from the model
            
        Returns:
            Dictionary with token usage information
        """
        pass
    
    @abc.abstractmethod
    def get_usage_statistics(self) -> Dict[str, Any]:
        """Get usage statistics.
        
        Returns:
            Dictionary with usage statistics
        """
        pass


class LLMServiceInterface(abc.ABC):
    """Interface for language model services."""
    
    @abc.abstractmethod
    def get_client(self, provider: str, model_name: str) -> LLMClientInterface:
        """Get a language model client.
        
        Args:
            provider: Provider name
            model_name: Model name
            
        Returns:
            Language model client
        """
        pass
    
    @abc.abstractmethod
    def complete(self, prompt: str, system_prompt: Optional[str] = None, 
              provider: Optional[str] = None, model_name: Optional[str] = None,
              **kwargs) -> str:
        """Get a completion from a model.
        
        Args:
            prompt: The prompt to send to the model
            system_prompt: Optional system prompt
            provider: Optional provider name (uses default if not specified)
            model_name: Optional model name (uses default if not specified)
            **kwargs: Additional parameters for the model
            
        Returns:
            The model's response
        """
        pass
    
    @abc.abstractmethod
    def evaluate(self, providers: List[str], models: List[str], 
               prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Evaluate multiple models on the same prompt.
        
        Args:
            providers: List of provider names
            models: List of model names
            prompt: The prompt to evaluate
            system_prompt: Optional system prompt
            
        Returns:
            Dictionary mapping model names to evaluation results
        """
        pass
    
    @abc.abstractmethod
    def get_token_estimate(self, text: str, provider: str, model_name: str) -> int:
        """Estimate the number of tokens in a text.
        
        Args:
            text: The text to estimate
            provider: Provider name
            model_name: Model name
            
        Returns:
            Estimated number of tokens
        """
        pass
    
    @abc.abstractmethod
    def get_cost_estimate(self, input_tokens: int, output_tokens: int, 
                       provider: str, model_name: str) -> float:
        """Estimate the cost of a request.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            provider: Provider name
            model_name: Model name
            
        Returns:
            Estimated cost in USD
        """
        pass


class LLMFactoryInterface(abc.ABC):
    """Interface for language model factories."""
    
    @abc.abstractmethod
    def create_client(self, provider: str, model_name: str, 
                    verify: bool = True) -> LLMClientInterface:
        """Create a language model client.
        
        Args:
            provider: Provider name
            model_name: Model name
            verify: Whether to verify model accuracy on initialization
            
        Returns:
            Language model client
        """
        pass
    
    @abc.abstractmethod
    def create_service(self, default_provider: Optional[str] = None,
                     default_model: Optional[str] = None) -> LLMServiceInterface:
        """Create a language model service.
        
        Args:
            default_provider: Default provider name
            default_model: Default model name
            
        Returns:
            Language model service
        """
        pass
    
    @abc.abstractmethod
    def get_available_providers(self) -> List[str]:
        """Get a list of available providers.
        
        Returns:
            List of provider names
        """
        pass
    
    @abc.abstractmethod
    def get_available_models(self, provider: str) -> List[str]:
        """Get a list of available models for a provider.
        
        Args:
            provider: Provider name
            
        Returns:
            List of model names
        """
        pass