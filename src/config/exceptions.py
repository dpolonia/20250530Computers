"""
Configuration exceptions module.

This module defines custom exceptions for the configuration system.
"""


class ConfigError(Exception):
    """Base class for all configuration exceptions."""
    pass


class ConfigValidationError(ConfigError):
    """Exception raised for configuration validation errors."""
    
    def __init__(self, message: str, errors: list = None):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            errors: List of specific validation errors
        """
        self.errors = errors or []
        super().__init__(message)


class ConfigFileError(ConfigError):
    """Exception raised for configuration file errors."""
    pass


class EnvVarError(ConfigError):
    """Exception raised for environment variable errors."""
    pass


class APIKeyError(ConfigError):
    """Exception raised for API key errors."""
    pass


class ModelError(ConfigError):
    """Exception raised for model-related errors."""
    pass


class BudgetError(ConfigError):
    """Exception raised for budget-related errors."""
    
    def __init__(self, message: str, current_cost: float, budget: float):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            current_cost: Current cost that exceeded the budget
            budget: Budget limit
        """
        self.current_cost = current_cost
        self.budget = budget
        super().__init__(message)


class MissingRequiredArgumentError(ConfigError):
    """Exception raised for missing required command-line arguments."""
    
    def __init__(self, argument_name: str):
        """
        Initialize the exception.
        
        Args:
            argument_name: Name of the missing argument
        """
        self.argument_name = argument_name
        super().__init__(f"Missing required argument: {argument_name}")


class InvalidArgumentError(ConfigError):
    """Exception raised for invalid command-line arguments."""
    
    def __init__(self, argument_name: str, value: str, reason: str = None):
        """
        Initialize the exception.
        
        Args:
            argument_name: Name of the invalid argument
            value: Invalid value
            reason: Reason why the value is invalid
        """
        self.argument_name = argument_name
        self.value = value
        self.reason = reason
        
        message = f"Invalid value for argument {argument_name}: {value}"
        if reason:
            message += f" ({reason})"
        
        super().__init__(message)