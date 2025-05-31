"""
Error handling module.

This module provides centralized error handling functionality for the paper revision tool,
including custom exceptions, error reporting, and recovery mechanisms.
"""

from src.errors.exceptions import (
    PaperRevisionError, LLMError, RequestError, ResponseError,
    TokenLimitError, FileError, ValidationError, 
    APIError, BudgetError, TimeoutError, RecoveryError,
    NetworkError, ResourceNotFoundError, PermissionError,
    parse_exception
)

from src.errors.handling import (
    ErrorHandler, ErrorContext, ErrorHandlerContext,
    get_error_handler, set_error_handler,
    handle_errors
)

from src.errors.reporting import (
    ErrorReporter, ConsoleReporter, FileReporter,
    JsonReporter, LoggingReporter, CompositeReporter,
    CallbackReporter, get_reporter, set_reporter,
    report_error, create_default_reporter
)

from src.errors.recovery import (
    RetryStrategy, register_recovery_strategy,
    get_recovery_strategies, retry_strategy,
    network_retry_strategy, llm_retry_strategy,
    token_limit_recovery_strategy, file_fallback_strategy
)

# Version of the error handling module
__version__ = "1.0.0"