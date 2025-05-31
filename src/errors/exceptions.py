"""
Custom exceptions for the paper revision tool.

This module defines a hierarchy of custom exceptions for the paper revision tool,
providing consistent error classification and handling.
"""

import sys
import traceback
from typing import Dict, Any, Optional, Type, List, Union


class PaperRevisionError(Exception):
    """Base class for all paper revision tool exceptions."""
    
    def __init__(
        self, 
        message: str, 
        cause: Optional[Exception] = None, 
        details: Optional[Dict[str, Any]] = None,
        recoverable: bool = True
    ):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            cause: Original exception that caused this error
            details: Additional details about the error
            recoverable: Whether the error is potentially recoverable
        """
        self.cause = cause
        self.details = details or {}
        self.recoverable = recoverable
        self.traceback = None
        
        # Capture traceback for debugging
        if cause:
            self.traceback = traceback.extract_tb(sys.exc_info()[2])
        
        # Format message with cause
        full_message = message
        if cause:
            full_message = f"{message} - Caused by: {type(cause).__name__}: {str(cause)}"
        
        super().__init__(full_message)
    
    def get_error_info(self) -> Dict[str, Any]:
        """
        Get structured information about the error.
        
        Returns:
            Dictionary with error information
        """
        return {
            "error_type": type(self).__name__,
            "message": str(self),
            "recoverable": self.recoverable,
            "details": self.details,
            "cause": {
                "type": type(self.cause).__name__,
                "message": str(self.cause)
            } if self.cause else None,
            "traceback": traceback.format_list(self.traceback) if self.traceback else None
        }


class LLMError(PaperRevisionError):
    """Exception raised for errors related to language models."""
    
    def __init__(
        self, 
        message: str, 
        provider: Optional[str] = None,
        model: Optional[str] = None,
        cause: Optional[Exception] = None, 
        details: Optional[Dict[str, Any]] = None,
        recoverable: bool = True
    ):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            provider: LLM provider name
            model: Model name
            cause: Original exception that caused this error
            details: Additional details about the error
            recoverable: Whether the error is potentially recoverable
        """
        details = details or {}
        details.update({
            "provider": provider,
            "model": model
        })
        super().__init__(message, cause, details, recoverable)


class RequestError(LLMError):
    """Exception raised for errors in requests to LLM APIs."""
    pass


class ResponseError(LLMError):
    """Exception raised for errors in responses from LLM APIs."""
    pass


class TokenLimitError(LLMError):
    """Exception raised when token limits are exceeded."""
    
    def __init__(
        self, 
        message: str, 
        provider: Optional[str] = None,
        model: Optional[str] = None,
        token_count: Optional[int] = None,
        token_limit: Optional[int] = None,
        cause: Optional[Exception] = None, 
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            provider: LLM provider name
            model: Model name
            token_count: Number of tokens in the request
            token_limit: Maximum allowed tokens
            cause: Original exception that caused this error
            details: Additional details about the error
        """
        details = details or {}
        details.update({
            "token_count": token_count,
            "token_limit": token_limit
        })
        super().__init__(message, provider, model, cause, details, recoverable=True)


class FileError(PaperRevisionError):
    """Exception raised for file-related errors."""
    
    def __init__(
        self, 
        message: str, 
        file_path: Optional[str] = None,
        operation: Optional[str] = None,
        cause: Optional[Exception] = None, 
        details: Optional[Dict[str, Any]] = None,
        recoverable: bool = True
    ):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            file_path: Path to the file
            operation: Operation being performed (read, write, etc.)
            cause: Original exception that caused this error
            details: Additional details about the error
            recoverable: Whether the error is potentially recoverable
        """
        details = details or {}
        details.update({
            "file_path": file_path,
            "operation": operation
        })
        super().__init__(message, cause, details, recoverable)


class ValidationError(PaperRevisionError):
    """Exception raised for validation errors."""
    
    def __init__(
        self, 
        message: str, 
        errors: Optional[List[str]] = None,
        cause: Optional[Exception] = None, 
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            errors: List of validation errors
            cause: Original exception that caused this error
            details: Additional details about the error
        """
        details = details or {}
        details["errors"] = errors or []
        super().__init__(message, cause, details, recoverable=True)


class APIError(PaperRevisionError):
    """Exception raised for API-related errors."""
    
    def __init__(
        self, 
        message: str, 
        api_name: Optional[str] = None,
        status_code: Optional[int] = None,
        cause: Optional[Exception] = None, 
        details: Optional[Dict[str, Any]] = None,
        recoverable: bool = True
    ):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            api_name: Name of the API
            status_code: HTTP status code
            cause: Original exception that caused this error
            details: Additional details about the error
            recoverable: Whether the error is potentially recoverable
        """
        details = details or {}
        details.update({
            "api_name": api_name,
            "status_code": status_code
        })
        super().__init__(message, cause, details, recoverable)


class BudgetError(PaperRevisionError):
    """Exception raised for budget-related errors."""
    
    def __init__(
        self, 
        message: str, 
        current_cost: Optional[float] = None,
        budget_limit: Optional[float] = None,
        cause: Optional[Exception] = None, 
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            current_cost: Current cost
            budget_limit: Budget limit
            cause: Original exception that caused this error
            details: Additional details about the error
        """
        details = details or {}
        details.update({
            "current_cost": current_cost,
            "budget_limit": budget_limit
        })
        super().__init__(message, cause, details, recoverable=False)


class TimeoutError(PaperRevisionError):
    """Exception raised for timeout errors."""
    
    def __init__(
        self, 
        message: str, 
        operation: Optional[str] = None,
        timeout: Optional[float] = None,
        cause: Optional[Exception] = None, 
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            operation: Operation that timed out
            timeout: Timeout value in seconds
            cause: Original exception that caused this error
            details: Additional details about the error
        """
        details = details or {}
        details.update({
            "operation": operation,
            "timeout": timeout
        })
        super().__init__(message, cause, details, recoverable=True)


class RecoveryError(PaperRevisionError):
    """Exception raised when error recovery fails."""
    
    def __init__(
        self, 
        message: str, 
        original_error: Optional[Exception] = None,
        recovery_strategy: Optional[str] = None,
        cause: Optional[Exception] = None, 
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            original_error: Original error that recovery was attempting to handle
            recovery_strategy: Name of the recovery strategy that failed
            cause: Exception that occurred during recovery
            details: Additional details about the error
        """
        details = details or {}
        details.update({
            "recovery_strategy": recovery_strategy,
            "original_error": str(original_error) if original_error else None
        })
        super().__init__(message, cause, details, recoverable=False)


class NetworkError(PaperRevisionError):
    """Exception raised for network-related errors."""
    pass


class ResourceNotFoundError(PaperRevisionError):
    """Exception raised when a resource is not found."""
    
    def __init__(
        self, 
        message: str, 
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        cause: Optional[Exception] = None, 
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            resource_type: Type of resource
            resource_id: ID of the resource
            cause: Original exception that caused this error
            details: Additional details about the error
        """
        details = details or {}
        details.update({
            "resource_type": resource_type,
            "resource_id": resource_id
        })
        super().__init__(message, cause, details, recoverable=False)


class PermissionError(PaperRevisionError):
    """Exception raised for permission-related errors."""
    
    def __init__(
        self, 
        message: str, 
        resource: Optional[str] = None,
        operation: Optional[str] = None,
        cause: Optional[Exception] = None, 
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            resource: Resource that was being accessed
            operation: Operation being performed
            cause: Original exception that caused this error
            details: Additional details about the error
        """
        details = details or {}
        details.update({
            "resource": resource,
            "operation": operation
        })
        super().__init__(message, cause, details, recoverable=False)


def parse_exception(exception: Exception) -> PaperRevisionError:
    """
    Parse a standard exception into a PaperRevisionError.
    
    Args:
        exception: Exception to parse
        
    Returns:
        Corresponding PaperRevisionError
    """
    # Already a PaperRevisionError
    if isinstance(exception, PaperRevisionError):
        return exception
    
    # Map standard exceptions to custom exceptions
    exception_map = {
        ValueError: ValidationError,
        TypeError: ValidationError,
        FileNotFoundError: FileError,
        PermissionError: PermissionError,
        TimeoutError: TimeoutError,
        ConnectionError: NetworkError,
        FileExistsError: FileError,
        IsADirectoryError: FileError,
        NotADirectoryError: FileError,
        OSError: FileError,
        IOError: FileError,
    }
    
    # Find matching exception type
    for exc_type, paper_exc_type in exception_map.items():
        if isinstance(exception, exc_type):
            return paper_exc_type(
                message=str(exception),
                cause=exception,
                details={"original_type": type(exception).__name__}
            )
    
    # Default to base PaperRevisionError
    return PaperRevisionError(
        message=str(exception),
        cause=exception,
        details={"original_type": type(exception).__name__}
    )