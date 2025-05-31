"""
LLM adapter implementation.

This module implements the LLM adapter interface, providing functionality for
interacting with language models.
"""

import logging
import os
import time
from typing import Dict, Any, Optional

from src.core.context import RevisionContext
from src.adapters.interfaces import LLMAdapterInterface


class LLMAdapter(LLMAdapterInterface):
    """
    Adapter for language model interactions.
    
    This adapter is responsible for handling interactions with language models,
    providing a consistent interface regardless of the underlying provider.
    """
    
    def __init__(self, context: RevisionContext):
        """
        Initialize the LLM adapter.
        
        Args:
            context: The shared revision context
        """
        self.context = context
        self.logger = context.logger or logging.getLogger(__name__)
        self.provider = context.provider
        self.model_name = context.model_name
        self.total_tokens_used = 0
        self.total_cost = 0.0
        
        # Ensure the client is initialized
        if not context.llm_client:
            context.setup_llm_client()
        
        self.llm_client = context.llm_client
    
    def get_completion(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Get a completion from the language model.
        
        Args:
            prompt: The prompt to send to the model
            system_prompt: Optional system prompt
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            The model's response
        """
        self.logger.debug(f"Getting completion from {self.provider}/{self.model_name}")
        
        # Implement retry logic for resilience
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                # Use the LLM client from the context
                response = self.llm_client.get_completion(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens
                )
                
                # Update token usage and cost tracking
                self.total_tokens_used += self.llm_client.total_tokens_used
                self.total_cost += self.llm_client.total_cost
                
                # Update context statistics
                self.context.process_statistics["total_tokens"] = self.context.process_statistics.get("total_tokens", 0) + self.llm_client.total_tokens_used
                self.context.process_statistics["total_cost"] = self.context.process_statistics.get("total_cost", 0.0) + self.llm_client.total_cost
                self.context.process_statistics["completion_requests"] = self.context.process_statistics.get("completion_requests", 0) + 1
                
                return response
                
            except Exception as e:
                self.logger.warning(f"Error in completion attempt {attempt+1}/{max_retries}: {e}")
                
                if attempt < max_retries - 1:
                    # Exponential backoff
                    wait_time = retry_delay * (2 ** attempt)
                    self.logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"Failed to get completion after {max_retries} attempts")
                    raise
    
    def get_token_estimate(self, text: str) -> int:
        """
        Get an estimate of the number of tokens in a text.
        
        Args:
            text: The text to estimate
            
        Returns:
            Estimated number of tokens
        """
        # Use the LLM client's tokenization method if available
        if hasattr(self.llm_client, 'get_token_count'):
            return self.llm_client.get_token_count(text)
        
        # Fallback estimation method: assume ~4 characters per token
        return len(text) // 4