"""
Error recovery strategies for the paper revision tool.

This module provides recovery strategies for various types of errors,
allowing the application to continue operation even in the presence of failures.
"""

import logging
import time
from typing import Dict, Any, Optional, Type, List, Union, Callable, TypeVar, Generic, cast

from .exceptions import (
    PaperRevisionError, 
    LLMError, 
    RequestError, 
    ResponseError,
    TokenLimitError,
    NetworkError,
    APIError,
    TimeoutError,
    FileError
)


# Type definitions
T = TypeVar('T')
RecoveryResult = TypeVar('RecoveryResult')
RecoveryStrategy = Callable[[PaperRevisionError, Dict[str, Any]], RecoveryResult]


class RetryStrategy:
    """
    Strategy for retrying operations that fail with recoverable errors.
    
    This strategy implements exponential backoff with jitter.
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
        jitter: float = 0.1,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the retry strategy.
        
        Args:
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay between retries in seconds
            max_delay: Maximum delay between retries in seconds
            backoff_factor: Factor by which to increase delay after each attempt
            jitter: Random jitter factor to add to delay
            logger: Logger to use for logging retry attempts
        """
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter
        self.logger = logger or logging.getLogger(__name__)
        
    def calculate_delay(self, attempt: int) -> float:
        """
        Calculate the delay for a retry attempt.
        
        Args:
            attempt: Current attempt number (0-based)
            
        Returns:
            Delay in seconds
        """
        import random
        
        # Calculate base delay with exponential backoff
        delay = min(
            self.max_delay,
            self.initial_delay * (self.backoff_factor ** attempt)
        )
        
        # Add jitter
        if self.jitter > 0:
            jitter_amount = delay * self.jitter
            delay += random.uniform(-jitter_amount, jitter_amount)
            
        return max(0, delay)
        
    def retry(
        self, 
        operation: Callable[[], T],
        error_types: Optional[List[Type[Exception]]] = None
    ) -> T:
        """
        Retry an operation with exponential backoff.
        
        Args:
            operation: Operation to retry
            error_types: Types of errors to retry on, or None for all
            
        Returns:
            Result of the operation
            
        Raises:
            Exception: If all retry attempts fail
        """
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return operation()
            except Exception as e:
                # Check if we should retry this error
                if error_types is not None:
                    retry_error = False
                    for error_type in error_types:
                        if isinstance(e, error_type):
                            retry_error = True
                            break
                    if not retry_error:
                        raise
                
                last_error = e
                
                # If this was the last attempt, re-raise
                if attempt >= self.max_retries:
                    raise
                
                # Calculate delay for next attempt
                delay = self.calculate_delay(attempt)
                
                # Log retry attempt
                self.logger.warning(
                    f"Retry attempt {attempt + 1}/{self.max_retries} after error: {str(e)}. "
                    f"Retrying in {delay:.2f} seconds."
                )
                
                # Wait before retrying
                time.sleep(delay)
        
        # This should never be reached, but just in case
        assert last_error is not None
        raise last_error


# Registry of recovery strategies
_strategies: Dict[Type[PaperRevisionError], List[RecoveryStrategy]] = {}
_global_strategies: List[RecoveryStrategy] = []


def register_recovery_strategy(
    strategy: RecoveryStrategy,
    error_types: Optional[List[Type[PaperRevisionError]]] = None
) -> None:
    """
    Register a recovery strategy.
    
    Args:
        strategy: Recovery strategy function
        error_types: List of error types to handle, or None for all types
    """
    global _strategies, _global_strategies
    
    if error_types is None:
        _global_strategies.append(strategy)
        return
        
    for error_type in error_types:
        if error_type not in _strategies:
            _strategies[error_type] = []
        _strategies[error_type].append(strategy)


def get_recovery_strategies(
    error: PaperRevisionError
) -> List[RecoveryStrategy]:
    """
    Get recovery strategies for an error.
    
    Args:
        error: Error to get strategies for
        
    Returns:
        List of applicable recovery strategies
    """
    global _strategies, _global_strategies
    
    error_type = type(error)
    strategies = []
    
    # Find strategies for this error type and its parent classes
    for strategy_type, strats in _strategies.items():
        if issubclass(error_type, strategy_type):
            strategies.extend(strats)
    
    # Add global strategies
    strategies.extend(_global_strategies)
    
    return strategies


# Predefined recovery strategies

def retry_strategy(
    error: PaperRevisionError,
    context: Dict[str, Any]
) -> Any:
    """
    Recovery strategy that retries the operation.
    
    Args:
        error: Error to recover from
        context: Error context
        
    Returns:
        Result of the retried operation
        
    Raises:
        Exception: If all retry attempts fail
    """
    # Get retry parameters from context
    max_retries = context.get("max_retries", 3)
    initial_delay = context.get("initial_delay", 1.0)
    max_delay = context.get("max_delay", 60.0)
    backoff_factor = context.get("backoff_factor", 2.0)
    jitter = context.get("jitter", 0.1)
    
    # Get the operation to retry
    operation = context.get("operation")
    if operation is None:
        raise ValueError("No operation provided in context")
    
    # Create retry strategy
    retry = RetryStrategy(
        max_retries=max_retries,
        initial_delay=initial_delay,
        max_delay=max_delay,
        backoff_factor=backoff_factor,
        jitter=jitter
    )
    
    # Get error types to retry on
    error_types = context.get("error_types")
    
    # Retry the operation
    return retry.retry(operation, error_types)


def network_retry_strategy(
    error: PaperRevisionError,
    context: Dict[str, Any]
) -> Any:
    """
    Recovery strategy for network errors that retries the operation.
    
    Args:
        error: Error to recover from
        context: Error context
        
    Returns:
        Result of the retried operation
        
    Raises:
        Exception: If all retry attempts fail
    """
    # Update context with network-specific retry parameters
    context.update({
        "max_retries": 5,
        "initial_delay": 2.0,
        "backoff_factor": 1.5,
        "error_types": [NetworkError, ConnectionError, TimeoutError]
    })
    
    # Delegate to general retry strategy
    return retry_strategy(error, context)


def llm_retry_strategy(
    error: PaperRevisionError,
    context: Dict[str, Any]
) -> Any:
    """
    Recovery strategy for LLM errors that retries the operation.
    
    Args:
        error: Error to recover from
        context: Error context
        
    Returns:
        Result of the retried operation
        
    Raises:
        Exception: If all retry attempts fail
    """
    # Update context with LLM-specific retry parameters
    context.update({
        "max_retries": 3,
        "initial_delay": 2.0,
        "backoff_factor": 2.0,
        "error_types": [RequestError, ResponseError, APIError]
    })
    
    # Delegate to general retry strategy
    return retry_strategy(error, context)


def token_limit_recovery_strategy(
    error: TokenLimitError,
    context: Dict[str, Any]
) -> Any:
    """
    Recovery strategy for token limit errors that splits the input.
    
    Args:
        error: Error to recover from
        context: Error context
        
    Returns:
        Result of the operation with split input
        
    Raises:
        Exception: If recovery fails
    """
    # Get the operation to retry
    operation = context.get("operation")
    if operation is None:
        raise ValueError("No operation provided in context")
    
    # Get the input that caused the token limit error
    input_text = context.get("input_text")
    if input_text is None:
        raise ValueError("No input_text provided in context")
    
    # Get the processor function that will process chunks
    processor = context.get("processor")
    if processor is None:
        raise ValueError("No processor function provided in context")
    
    # Split the input into chunks
    chunks = split_text_into_chunks(input_text, error.details.get("token_limit", 2000))
    
    # Process each chunk
    results = []
    for chunk in chunks:
        # Update context with the current chunk
        chunk_context = context.copy()
        chunk_context["input_text"] = chunk
        
        # Process the chunk
        result = processor(chunk)
        results.append(result)
    
    # Combine the results
    combiner = context.get("combiner")
    if combiner is None:
        # Default combiner just returns the list of results
        return results
    
    # Use the provided combiner function
    return combiner(results)


def split_text_into_chunks(text: str, max_tokens: int) -> List[str]:
    """
    Split text into chunks that fit within the token limit.
    
    This is a simple implementation that splits by paragraphs and then by sentences
    if needed. For more accurate token counting, a tokenizer for the specific
    LLM should be used.
    
    Args:
        text: Text to split
        max_tokens: Maximum tokens per chunk
        
    Returns:
        List of text chunks
    """
    import re
    
    # Simple approximation: 1 token is roughly 4 characters
    max_chars = max_tokens * 4
    
    # First, split by paragraphs
    paragraphs = re.split(r'\n\s*\n', text)
    
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        # If the paragraph itself is too long, split by sentences
        if len(paragraph) > max_chars:
            sentences = re.split(r'(?<=[.!?])\s+', paragraph)
            for sentence in sentences:
                if len(current_chunk) + len(sentence) + 2 <= max_chars:
                    if current_chunk:
                        current_chunk += "\n\n"
                    current_chunk += sentence
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = sentence
        else:
            # Check if adding this paragraph exceeds the limit
            if len(current_chunk) + len(paragraph) + 2 <= max_chars:
                if current_chunk:
                    current_chunk += "\n\n"
                current_chunk += paragraph
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = paragraph
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks


def file_fallback_strategy(
    error: FileError,
    context: Dict[str, Any]
) -> Any:
    """
    Recovery strategy for file errors that tries alternative file paths.
    
    Args:
        error: Error to recover from
        context: Error context
        
    Returns:
        Result of the operation with alternative file path
        
    Raises:
        Exception: If all alternatives fail
    """
    # Get the operation to retry
    operation = context.get("operation")
    if operation is None:
        raise ValueError("No operation provided in context")
    
    # Get alternative file paths
    alternative_paths = context.get("alternative_paths")
    if not alternative_paths:
        raise ValueError("No alternative_paths provided in context")
    
    # Get original file path
    original_path = error.details.get("file_path")
    if not original_path:
        raise ValueError("No file_path in error details")
    
    # Try each alternative path
    for path in alternative_paths:
        try:
            # Replace the file path in the context
            path_context = context.copy()
            path_context["file_path"] = path
            
            # Call the operation with the new path
            return operation(path_context)
        except Exception as e:
            # Log the error but continue with next alternative
            logging.warning(f"Alternative path {path} failed: {str(e)}")
    
    # If all alternatives fail, raise the original error
    raise error


# Register predefined recovery strategies
register_recovery_strategy(network_retry_strategy, [NetworkError, TimeoutError])
register_recovery_strategy(llm_retry_strategy, [RequestError, ResponseError, APIError])
register_recovery_strategy(token_limit_recovery_strategy, [TokenLimitError])
register_recovery_strategy(file_fallback_strategy, [FileError])