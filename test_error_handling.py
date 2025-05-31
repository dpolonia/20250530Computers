#!/usr/bin/env python3
"""
Test script for the error handling system.

This script tests the error handling system with various error scenarios
to ensure proper error handling, reporting, and recovery.
"""

import logging
import os
import sys
import tempfile
from typing import Dict, Any, List, Optional

from src.errors import (
    PaperRevisionError, LLMError, FileError, ValidationError,
    TokenLimitError, NetworkError, APIError, TimeoutError, RecoveryError,
    ErrorHandler, ErrorContext, ErrorHandlerContext,
    ConsoleReporter, FileReporter, LoggingReporter, CompositeReporter,
    RetryStrategy, handle_errors, report_error, create_default_reporter,
    get_reporter, set_reporter
)


def test_basic_exception_handling():
    """Test basic exception handling."""
    print("\n=== Testing basic exception handling ===")
    
    # Set up a console reporter
    set_reporter(ConsoleReporter())
    
    # Define a function that raises an error
    @handle_errors(recover=False)
    def function_with_error():
        raise FileError(
            message="Could not read file",
            file_path="/path/to/nonexistent/file.txt",
            operation="read"
        )
    
    # Call the function and handle the error
    try:
        function_with_error()
        print("Function completed without error (unexpected)")
    except PaperRevisionError as e:
        print(f"Caught expected error: {type(e).__name__}")
    
    print("Basic exception handling test completed")


def test_error_recovery():
    """Test error recovery."""
    print("\n=== Testing error recovery ===")
    
    # Define a recovery function
    def recovery_function(error, context):
        print(f"Recovering from {type(error).__name__}")
        return "Recovery result"
    
    # Set up error handler with recovery
    handler = ErrorHandler()
    handler.register_recovery_strategy(recovery_function, [LLMError])
    
    # Use the handler
    try:
        # Create an error
        error = LLMError(
            message="API request failed",
            provider="anthropic",
            model="claude-3-opus-20240229"
        )
        
        # Handle and recover from the error
        result = handler.recover_from_error(error, {"test": "context"})
        print(f"Recovery result: {result}")
        
    except RecoveryError as e:
        print(f"Recovery failed: {e}")
    
    print("Error recovery test completed")


def test_retry_strategy():
    """Test retry strategy."""
    print("\n=== Testing retry strategy ===")
    
    # Set up retry strategy
    retry = RetryStrategy(
        max_retries=3,
        initial_delay=0.1,  # Short delay for testing
        backoff_factor=2.0,
        jitter=0.1
    )
    
    # Define a function that succeeds on the third attempt
    attempt = [0]
    
    def operation():
        attempt[0] += 1
        print(f"Attempt {attempt[0]}")
        if attempt[0] < 3:
            raise NetworkError(
                message=f"Connection failed on attempt {attempt[0]}",
                recoverable=True
            )
        return f"Success on attempt {attempt[0]}"
    
    # Retry the operation
    try:
        result = retry.retry(operation)
        print(f"Retry result: {result}")
    except PaperRevisionError as e:
        print(f"Retry failed: {e}")
    
    print("Retry strategy test completed")


def test_error_context_manager():
    """Test error context manager."""
    print("\n=== Testing error context manager ===")
    
    # Create a context with recovery
    with ErrorHandlerContext(
        recover=True, 
        default_value="Default recovery value",
        context={"test": "context"}
    ) as context:
        # Raise an error inside the context
        raise TokenLimitError(
            message="Token limit exceeded",
            provider="anthropic",
            model="claude-3-opus-20240229",
            token_count=100000,
            token_limit=10000
        )
    
    # If we get here, recovery succeeded
    print(f"Context result: {context.result}")
    print("Error context manager test completed")


def test_error_reporting():
    """Test error reporting."""
    print("\n=== Testing error reporting ===")
    
    # Create a temporary log file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".log") as temp:
        log_path = temp.name
    
    try:
        # Set up reporters
        console_reporter = ConsoleReporter(show_traceback=False)
        file_reporter = FileReporter(log_path, include_timestamp=True)
        
        # Create a composite reporter
        reporter = CompositeReporter([console_reporter, file_reporter])
        set_reporter(reporter)
        
        # Create an error
        error = APIError(
            message="API request failed with status code 500",
            api_name="OpenAI API",
            status_code=500,
            details={"endpoint": "/v1/chat/completions"}
        )
        
        # Report the error
        report_error(error, {"test_name": "error_reporting_test"})
        
        # Check if the log file was created and contains error information
        with open(log_path, "r") as f:
            log_content = f.read()
            print(f"Log file contains {len(log_content)} characters")
            if "API request failed" in log_content:
                print("Error information correctly written to log file")
            else:
                print("Error: Log file does not contain expected error information")
    
    finally:
        # Clean up
        if os.path.exists(log_path):
            os.unlink(log_path)
    
    print("Error reporting test completed")


def test_decorator_with_recovery():
    """Test decorator with recovery."""
    print("\n=== Testing decorator with recovery ===")
    
    # Define a function with error handling and recovery
    @handle_errors(recover=True, default_value="Default recovery value")
    def function_with_recovery():
        raise TimeoutError(
            message="Operation timed out",
            operation="API request",
            timeout=30.0
        )
    
    # Call the function
    result = function_with_recovery()
    print(f"Function result after recovery: {result}")
    
    print("Decorator with recovery test completed")


def test_parse_exception():
    """Test parsing standard exceptions to PaperRevisionError."""
    print("\n=== Testing parse_exception ===")
    
    from src.errors.exceptions import parse_exception
    
    # Test with different standard exceptions
    exceptions = [
        ValueError("Invalid value"),
        FileNotFoundError("File not found"),
        PermissionError("Permission denied"),
        TimeoutError("Operation timed out"),
        ConnectionError("Connection failed"),
        TypeError("Invalid type")
    ]
    
    for original in exceptions:
        # Parse the exception
        paper_error = parse_exception(original)
        
        # Check the result
        print(f"Original: {type(original).__name__}, Parsed: {type(paper_error).__name__}")
        if isinstance(paper_error, PaperRevisionError):
            print(f"  Message: {paper_error}")
            print(f"  Original type preserved: {paper_error.details.get('original_type')}")
        else:
            print(f"  Error: Parsing failed, result is not a PaperRevisionError")
    
    print("Parse exception test completed")


def main():
    """Main function to run all tests."""
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    print("====================================")
    print("Error Handling System Integration Test")
    print("====================================\n")
    
    # Run all tests
    test_basic_exception_handling()
    test_error_recovery()
    test_retry_strategy()
    test_error_context_manager()
    test_error_reporting()
    test_decorator_with_recovery()
    test_parse_exception()
    
    print("\n====================================")
    print("All tests completed")
    print("====================================")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())