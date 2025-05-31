"""
Language Model factory for the Paper Revision System.

This module provides factories for creating language model clients and services.
"""

import logging
from typing import Dict, List, Optional, Type, Any

from src.interfaces.llm import (
    LLMClientInterface,
    LLMServiceInterface,
    LLMFactoryInterface
)

# Import concrete implementations
from src.utils.llm_client import (
    AnthropicClient,
    OpenAIClient,
    GeminiClient,
    BaseLLMClient
)

from src.services.llm_service import LLMService

# Configure logging
logger = logging.getLogger(__name__)


class LLMFactory(LLMFactoryInterface):
    """Factory for language model components."""
    
    def __init__(self):
        """Initialize the factory."""
        self._client_registry = {}
        self._service_registry = {}
        self._model_registry = {}
        
        # Register default client implementations
        self.register_client("anthropic", AnthropicClient)
        self.register_client("openai", OpenAIClient)
        self.register_client("google", GeminiClient)
        
        # Register default service implementation
        self.register_service("default", LLMService)
        
        # Register default models
        self._register_default_models()
    
    def register_client(self, provider: str, client_class: Type[LLMClientInterface]):
        """Register a language model client for a provider.
        
        Args:
            provider: Provider name
            client_class: Client class
        """
        self._client_registry[provider.lower()] = client_class
        logger.debug(f"Registered client for {provider}: {client_class.__name__}")
    
    def register_service(self, name: str, service_class: Type[LLMServiceInterface]):
        """Register a language model service.
        
        Args:
            name: Service name
            service_class: Service class
        """
        self._service_registry[name.lower()] = service_class
        logger.debug(f"Registered service: {name} - {service_class.__name__}")
    
    def register_model(self, provider: str, model_name: str, 
                     description: str, capabilities: Dict[str, Any]):
        """Register a model with its capabilities.
        
        Args:
            provider: Provider name
            model_name: Model name
            description: Model description
            capabilities: Dictionary of model capabilities
        """
        if provider.lower() not in self._model_registry:
            self._model_registry[provider.lower()] = {}
            
        self._model_registry[provider.lower()][model_name] = {
            "description": description,
            "capabilities": capabilities
        }
        
        logger.debug(f"Registered model: {provider}/{model_name}")
    
    def _register_default_models(self):
        """Register default models for each provider."""
        # Anthropic models
        self.register_model(
            "anthropic",
            "claude-3-opus-20240229",
            "Claude 3 Opus - Anthropic's most advanced model for highly complex tasks",
            {
                "context_window": 200000,
                "max_tokens": 4096,
                "price_per_1k_input": 0.015,
                "price_per_1k_output": 0.075,
                "capabilities": ["reasoning", "creative", "coding", "factual"]
            }
        )
        
        self.register_model(
            "anthropic",
            "claude-3-sonnet-20240229",
            "Claude 3 Sonnet - Balanced model for a wide range of tasks",
            {
                "context_window": 200000,
                "max_tokens": 4096,
                "price_per_1k_input": 0.003,
                "price_per_1k_output": 0.015,
                "capabilities": ["reasoning", "creative", "coding", "factual"]
            }
        )
        
        self.register_model(
            "anthropic",
            "claude-3-haiku-20240307",
            "Claude 3 Haiku - Fast and cost-effective model",
            {
                "context_window": 200000,
                "max_tokens": 4096,
                "price_per_1k_input": 0.00025,
                "price_per_1k_output": 0.00125,
                "capabilities": ["reasoning", "creative", "coding", "factual"]
            }
        )
        
        # OpenAI models
        self.register_model(
            "openai",
            "gpt-4o",
            "GPT-4o - OpenAI's most capable model for a wide range of tasks",
            {
                "context_window": 128000,
                "max_tokens": 4096,
                "price_per_1k_input": 0.005,
                "price_per_1k_output": 0.015,
                "capabilities": ["reasoning", "creative", "coding", "factual"]
            }
        )
        
        self.register_model(
            "openai",
            "gpt-4-turbo",
            "GPT-4 Turbo - Optimized performance and cost",
            {
                "context_window": 128000,
                "max_tokens": 4096,
                "price_per_1k_input": 0.01,
                "price_per_1k_output": 0.03,
                "capabilities": ["reasoning", "creative", "coding", "factual"]
            }
        )
        
        self.register_model(
            "openai",
            "gpt-3.5-turbo",
            "GPT-3.5 Turbo - Fast and cost-effective model",
            {
                "context_window": 16000,
                "max_tokens": 4096,
                "price_per_1k_input": 0.0005,
                "price_per_1k_output": 0.0015,
                "capabilities": ["reasoning", "creative", "coding", "factual"]
            }
        )
        
        # Google models
        self.register_model(
            "google",
            "gemini-1.5-pro",
            "Gemini 1.5 Pro - Google's advanced multimodal model",
            {
                "context_window": 1000000,
                "max_tokens": 8192,
                "price_per_1k_input": 0.0025,
                "price_per_1k_output": 0.0025,
                "capabilities": ["reasoning", "creative", "coding", "factual", "multimodal"]
            }
        )
        
        self.register_model(
            "google",
            "gemini-1.0-pro",
            "Gemini 1.0 Pro - Balanced model for various tasks",
            {
                "context_window": 32000,
                "max_tokens": 8192,
                "price_per_1k_input": 0.00125,
                "price_per_1k_output": 0.00125,
                "capabilities": ["reasoning", "creative", "coding", "factual"]
            }
        )
    
    def create_client(self, provider: str, model_name: str, 
                    verify: bool = True) -> LLMClientInterface:
        """Create a language model client.
        
        Args:
            provider: Provider name
            model_name: Model name
            verify: Whether to verify model accuracy on initialization
            
        Returns:
            Language model client
            
        Raises:
            ValueError: If the provider is not supported
        """
        # Normalize provider name
        provider = provider.lower()
        
        # Check if provider is supported
        if provider not in self._client_registry:
            raise ValueError(f"Unsupported provider: {provider}")
        
        # Create client
        client_class = self._client_registry[provider]
        logger.debug(f"Creating client for {provider}/{model_name}")
        
        try:
            client = client_class(model_name)
            
            # Verify the client if requested
            if verify and hasattr(client, 'verify_model_accuracy'):
                client.verify_model_accuracy()
                
            return client
        except Exception as e:
            logger.error(f"Failed to create client for {provider}/{model_name}: {e}")
            raise
    
    def create_service(self, default_provider: Optional[str] = None,
                     default_model: Optional[str] = None) -> LLMServiceInterface:
        """Create a language model service.
        
        Args:
            default_provider: Default provider name
            default_model: Default model name
            
        Returns:
            Language model service
            
        Raises:
            ValueError: If no service is registered
        """
        # Check if any services are registered
        if not self._service_registry:
            raise ValueError("No language model services registered")
        
        # Use the default service
        service_name = "default"
        service_class = self._service_registry[service_name]
        logger.debug(f"Creating service: {service_name}")
        
        try:
            # Create a factory instance to pass to the service
            factory = self
            
            # Create the service
            service = service_class(factory, default_provider, default_model)
            return service
        except Exception as e:
            logger.error(f"Failed to create service: {e}")
            raise
    
    def get_available_providers(self) -> List[str]:
        """Get a list of available providers.
        
        Returns:
            List of provider names
        """
        return list(self._client_registry.keys())
    
    def get_available_models(self, provider: str) -> List[str]:
        """Get a list of available models for a provider.
        
        Args:
            provider: Provider name
            
        Returns:
            List of model names
            
        Raises:
            ValueError: If the provider is not supported
        """
        # Normalize provider name
        provider = provider.lower()
        
        # Check if provider is supported
        if provider not in self._model_registry:
            raise ValueError(f"Unsupported provider: {provider}")
            
        return list(self._model_registry[provider].keys())
    
    def get_model_info(self, provider: str, model_name: str) -> Dict[str, Any]:
        """Get information about a model.
        
        Args:
            provider: Provider name
            model_name: Model name
            
        Returns:
            Model information
            
        Raises:
            ValueError: If the model is not found
        """
        # Normalize provider name
        provider = provider.lower()
        
        # Check if provider is supported
        if provider not in self._model_registry:
            raise ValueError(f"Unsupported provider: {provider}")
            
        # Check if model is registered
        if model_name not in self._model_registry[provider]:
            raise ValueError(f"Unknown model: {provider}/{model_name}")
            
        return self._model_registry[provider][model_name]


# Create a singleton instance
_llm_factory = None

def get_llm_factory() -> LLMFactory:
    """Get the LLM factory singleton.
    
    Returns:
        LLM factory instance
    """
    global _llm_factory
    
    if _llm_factory is None:
        _llm_factory = LLMFactory()
        
    return _llm_factory