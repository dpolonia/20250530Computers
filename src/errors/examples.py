"""
Examples of using the error handling system.

This module provides examples of how to use the error handling system
in different scenarios.
"""

import logging
import time
from typing import Dict, Any, List, Optional

from src.errors import (
    PaperRevisionError, LLMError, FileError, TokenLimitError,
    ErrorHandler, ErrorHandlerContext, handle_errors,
    ConsoleReporter, FileReporter, CompositeReporter,
    RetryStrategy, register_recovery_strategy
)


# Example 1: Basic error handling with decorator

@handle_errors(recover=False)
def example_function_with_error():
    """Example function that raises an error."""
    raise FileError("Could not read file", file_path="/path/to/file", operation="read")


# Example 2: Error handling with recovery

@handle_errors(recover=True, default_value="Default Value")
def example_function_with_recovery():
    """Example function that raises a recoverable error."""
    raise TokenLimitError(
        message="Token limit exceeded",
        provider="anthropic",
        model="claude-3-opus-20240229",
        token_count=100000,
        token_limit=10000
    )


# Example 3: Using context manager for error handling

def example_with_context_manager():
    """Example function using context manager for error handling."""
    with ErrorHandlerContext(recover=True, default_value="Default from context") as context:
        # Code that might raise an error
        raise LLMError(
            message="LLM request failed",
            provider="anthropic",
            model="claude-3-opus-20240229"
        )
    
    # This will only execute if recovery succeeded
    return context.result


# Example 4: Custom recovery strategy

def custom_recovery_strategy(error: PaperRevisionError, context: Dict[str, Any]) -> str:
    """
    Custom recovery strategy that returns a predefined value.
    
    Args:
        error: Error to recover from
        context: Error context
        
    Returns:
        Predefined recovery value
    """
    return "Recovered with custom strategy"


# Example 5: Retry with exponential backoff

def example_retry_with_backoff():
    """Example function using retry with exponential backoff."""
    retry = RetryStrategy(
        max_retries=3,
        initial_delay=1.0,
        backoff_factor=2.0,
        jitter=0.1
    )
    
    # Define a function that fails the first two times
    attempt = [0]
    
    def operation():
        attempt[0] += 1
        if attempt[0] <= 2:
            raise ConnectionError(f"Connection failed on attempt {attempt[0]}")
        return f"Success on attempt {attempt[0]}"
    
    # Retry the operation
    return retry.retry(operation)


# Example 6: Complex error handling with multiple strategies

class ComplexExample:
    """Class demonstrating complex error handling."""
    
    def __init__(self):
        """Initialize the example."""
        self.logger = logging.getLogger(__name__)
        
        # Set up error handler
        self.error_handler = ErrorHandler(self.logger)
        
        # Set up reporters
        console_reporter = ConsoleReporter(show_traceback=True)
        file_reporter = FileReporter("errors.log", include_timestamp=True)
        
        # Create composite reporter
        self.reporter = CompositeReporter([console_reporter, file_reporter])
        
        # Register recovery strategy
        register_recovery_strategy(custom_recovery_strategy, [LLMError])
        
    def process_with_error_handling(self, input_data: str) -> Dict[str, Any]:
        """
        Process input data with comprehensive error handling.
        
        Args:
            input_data: Input data to process
            
        Returns:
            Processing result
        """
        result = {
            "success": True,
            "data": None,
            "errors": []
        }
        
        try:
            # Process the data
            result["data"] = self._process_data(input_data)
        except PaperRevisionError as e:
            # Handle the error
            result["success"] = False
            result["errors"].append(e.get_error_info())
            
            # Report the error
            self.reporter.report(e, {"input_data": input_data})
            
            # Try to recover
            if e.recoverable:
                try:
                    # Attempt recovery
                    recovery_result = self.error_handler.recover_from_error(
                        e, {"input_data": input_data}
                    )
                    
                    # Update result with recovery data
                    result["data"] = recovery_result
                    result["recovered"] = True
                except Exception as recovery_error:
                    # Recovery failed
                    result["recovered"] = False
                    result["recovery_error"] = str(recovery_error)
        
        return result
    
    def _process_data(self, input_data: str) -> Dict[str, Any]:
        """
        Process input data.
        
        Args:
            input_data: Input data to process
            
        Returns:
            Processed data
            
        Raises:
            LLMError: If processing fails
        """
        # Simulate processing error for demonstration
        if "error" in input_data.lower():
            raise LLMError(
                message="Processing failed",
                provider="anthropic",
                model="claude-3-opus-20240229"
            )
        
        # Return processed data
        return {
            "processed": input_data.upper(),
            "timestamp": time.time()
        }


# Example 7: Main function demonstrating all examples

def main():
    """Main function demonstrating error handling examples."""
    logging.basicConfig(level=logging.INFO)
    
    print("\n=== Example 1: Basic error handling with decorator ===")
    try:
        example_function_with_error()
    except PaperRevisionError as e:
        print(f"Caught error: {type(e).__name__}: {e}")
    
    print("\n=== Example 2: Error handling with recovery ===")
    result = example_function_with_recovery()
    print(f"Result after recovery: {result}")
    
    print("\n=== Example 3: Using context manager for error handling ===")
    result = example_with_context_manager()
    print(f"Result from context manager: {result}")
    
    print("\n=== Example 4: Retry with exponential backoff ===")
    result = example_retry_with_backoff()
    print(f"Result after retry: {result}")
    
    print("\n=== Example 5: Complex error handling with multiple strategies ===")
    example = ComplexExample()
    
    # Process with success
    result = example.process_with_error_handling("hello world")
    print(f"Success result: {result}")
    
    # Process with error and recovery
    result = example.process_with_error_handling("trigger error")
    print(f"Error result with recovery: {result}")
    
    print("\nAll examples completed successfully!")


if __name__ == "__main__":
    main()