"""
Error handling mechanisms for the paper revision tool.

This module provides a centralized error handling system with support for
registering handlers, formatting errors, and applying recovery strategies.
"""

import logging
import sys
import traceback
from typing import Dict, Any, Optional, Type, List, Union, Callable, TypeVar, Generic
from functools import wraps

from .exceptions import PaperRevisionError, RecoveryError


# Type definitions
T = TypeVar('T')
ErrorHandler = Callable[[PaperRevisionError, Dict[str, Any]], None]
RecoveryStrategy = Callable[[PaperRevisionError, Dict[str, Any]], Any]


class ErrorContext:
    """Context information for error handling."""
    
    def __init__(
        self,
        function_name: str,
        args: tuple = (),
        kwargs: Dict[str, Any] = None,
        context: Dict[str, Any] = None
    ):
        """
        Initialize the error context.
        
        Args:
            function_name: Name of the function where the error occurred
            args: Positional arguments to the function
            kwargs: Keyword arguments to the function
            context: Additional context information
        """
        self.function_name = function_name
        self.args = args
        self.kwargs = kwargs or {}
        self.context = context or {}
        self.timestamp = None  # Will be set when an error occurs
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the context to a dictionary.
        
        Returns:
            Dictionary representation of the context
        """
        return {
            "function_name": self.function_name,
            "args": self.args,
            "kwargs": self.kwargs,
            "context": self.context,
            "timestamp": self.timestamp
        }


class ErrorHandler:
    """
    Central error handler for the paper revision tool.
    
    This class provides a registry of error handlers and recovery strategies,
    and coordinates the error handling process.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the error handler.
        
        Args:
            logger: Logger to use for error logging
        """
        self.logger = logger or logging.getLogger(__name__)
        self.handlers: Dict[Type[PaperRevisionError], List[ErrorHandler]] = {}
        self.recovery_strategies: Dict[Type[PaperRevisionError], List[RecoveryStrategy]] = {}
        self.global_handlers: List[ErrorHandler] = []
        self.global_recovery_strategies: List[RecoveryStrategy] = []
        
    def register_handler(
        self, 
        handler: ErrorHandler, 
        error_types: Optional[List[Type[PaperRevisionError]]] = None
    ) -> None:
        """
        Register an error handler.
        
        Args:
            handler: Error handler function
            error_types: List of error types to handle, or None for all types
        """
        if error_types is None:
            self.global_handlers.append(handler)
            return
            
        for error_type in error_types:
            if error_type not in self.handlers:
                self.handlers[error_type] = []
            self.handlers[error_type].append(handler)
    
    def register_recovery_strategy(
        self, 
        strategy: RecoveryStrategy, 
        error_types: Optional[List[Type[PaperRevisionError]]] = None
    ) -> None:
        """
        Register a recovery strategy.
        
        Args:
            strategy: Recovery strategy function
            error_types: List of error types to handle, or None for all types
        """
        if error_types is None:
            self.global_recovery_strategies.append(strategy)
            return
            
        for error_type in error_types:
            if error_type not in self.recovery_strategies:
                self.recovery_strategies[error_type] = []
            self.recovery_strategies[error_type].append(strategy)
    
    def handle_error(
        self, 
        error: PaperRevisionError, 
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Handle an error using registered handlers.
        
        Args:
            error: Error to handle
            context: Error context
        """
        context = context or {}
        
        # Log the error
        self.logger.error(
            f"Error occurred: {type(error).__name__}: {str(error)}",
            extra={"error_info": error.get_error_info(), "context": context}
        )
        
        # Call specific handlers for this error type
        error_type = type(error)
        specific_handlers = []
        
        # Find handlers for this error type and its parent classes
        for handler_type, handlers in self.handlers.items():
            if issubclass(error_type, handler_type):
                specific_handlers.extend(handlers)
        
        # Call specific handlers
        for handler in specific_handlers:
            try:
                handler(error, context)
            except Exception as e:
                self.logger.error(
                    f"Error in handler {handler.__name__}: {str(e)}",
                    exc_info=True
                )
        
        # Call global handlers
        for handler in self.global_handlers:
            try:
                handler(error, context)
            except Exception as e:
                self.logger.error(
                    f"Error in global handler {handler.__name__}: {str(e)}",
                    exc_info=True
                )
    
    def recover_from_error(
        self, 
        error: PaperRevisionError, 
        context: Optional[Dict[str, Any]] = None,
        default_value: Optional[Any] = None
    ) -> Any:
        """
        Attempt to recover from an error using registered recovery strategies.
        
        Args:
            error: Error to recover from
            context: Error context
            default_value: Default value to return if recovery fails
            
        Returns:
            Result of the recovery strategy, or default_value if recovery fails
            
        Raises:
            RecoveryError: If recovery fails and no default value is provided
        """
        if not error.recoverable:
            # If the error is not recoverable, don't attempt recovery
            raise RecoveryError(
                message=f"Cannot recover from non-recoverable error: {type(error).__name__}",
                original_error=error,
                details=context
            )
        
        context = context or {}
        
        # Find recovery strategies for this error type
        error_type = type(error)
        specific_strategies = []
        
        # Find strategies for this error type and its parent classes
        for strategy_type, strategies in self.recovery_strategies.items():
            if issubclass(error_type, strategy_type):
                specific_strategies.extend(strategies)
        
        # Try specific strategies
        for strategy in specific_strategies:
            try:
                return strategy(error, context)
            except Exception as e:
                self.logger.warning(
                    f"Recovery strategy {strategy.__name__} failed: {str(e)}",
                    exc_info=True
                )
        
        # Try global strategies
        for strategy in self.global_recovery_strategies:
            try:
                return strategy(error, context)
            except Exception as e:
                self.logger.warning(
                    f"Global recovery strategy {strategy.__name__} failed: {str(e)}",
                    exc_info=True
                )
        
        # If we have a default value, return it
        if default_value is not None:
            return default_value
            
        # If all recovery strategies fail, raise a RecoveryError
        raise RecoveryError(
            message=f"All recovery strategies failed for error: {type(error).__name__}",
            original_error=error,
            details=context
        )


# Global error handler instance
_error_handler = None


def get_error_handler() -> ErrorHandler:
    """
    Get the global error handler instance.
    
    Returns:
        Global error handler
    """
    global _error_handler
    if _error_handler is None:
        _error_handler = ErrorHandler()
    return _error_handler


def set_error_handler(handler: ErrorHandler) -> None:
    """
    Set the global error handler instance.
    
    Args:
        handler: Error handler to use
    """
    global _error_handler
    _error_handler = handler


def handle_errors(
    error_types: Optional[List[Type[PaperRevisionError]]] = None,
    recover: bool = False,
    default_value: Optional[Any] = None
):
    """
    Decorator for handling errors in functions.
    
    Args:
        error_types: Types of errors to handle, or None for all
        recover: Whether to attempt recovery
        default_value: Default value to return if recovery fails
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            handler = get_error_handler()
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Convert to PaperRevisionError if needed
                if not isinstance(e, PaperRevisionError):
                    from .exceptions import parse_exception
                    error = parse_exception(e)
                else:
                    error = e
                
                # Create context
                context = ErrorContext(
                    function_name=func.__name__,
                    args=args,
                    kwargs=kwargs
                ).to_dict()
                
                # Handle the error
                handler.handle_error(error, context)
                
                # Attempt recovery if requested
                if recover:
                    return handler.recover_from_error(error, context, default_value)
                
                # Re-raise the error if not recovering
                raise
                
        return wrapper
    return decorator


class ErrorHandlerContext:
    """Context manager for handling errors."""
    
    def __init__(
        self, 
        error_types: Optional[List[Type[PaperRevisionError]]] = None,
        recover: bool = False,
        default_value: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the context manager.
        
        Args:
            error_types: Types of errors to handle, or None for all
            recover: Whether to attempt recovery
            default_value: Default value to return if recovery fails
            context: Additional context information
        """
        self.error_types = error_types
        self.recover = recover
        self.default_value = default_value
        self.context = context or {}
        self.handler = get_error_handler()
        
    def __enter__(self):
        """Enter the context manager."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the context manager.
        
        Args:
            exc_type: Exception type
            exc_val: Exception value
            exc_tb: Exception traceback
            
        Returns:
            True if the exception was handled, False otherwise
        """
        if exc_val is None:
            return False
            
        # Check if we should handle this error
        if self.error_types is not None:
            handled = False
            for error_type in self.error_types:
                if isinstance(exc_val, error_type):
                    handled = True
                    break
            if not handled:
                return False
        
        # Convert to PaperRevisionError if needed
        if not isinstance(exc_val, PaperRevisionError):
            from .exceptions import parse_exception
            error = parse_exception(exc_val)
        else:
            error = exc_val
        
        # Handle the error
        self.handler.handle_error(error, self.context)
        
        # Attempt recovery if requested
        if self.recover:
            try:
                self.recovery_result = self.handler.recover_from_error(
                    error, self.context, self.default_value
                )
                return True
            except RecoveryError:
                # If recovery fails, re-raise the original error
                return False
        
        # If not recovering, propagate the exception
        return False
    
    @property
    def result(self):
        """Get the result of the recovery."""
        if hasattr(self, 'recovery_result'):
            return self.recovery_result
        return None