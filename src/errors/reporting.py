"""
Error reporting mechanisms for the paper revision tool.

This module provides mechanisms for reporting errors to various outputs,
including console, files, and external services.
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, Any, Optional, Type, List, Union, TextIO, Callable

from .exceptions import PaperRevisionError


class ErrorReporter:
    """
    Base class for error reporters.
    
    Error reporters are responsible for formatting and outputting
    error information to various destinations.
    """
    
    def report(
        self, 
        error: PaperRevisionError, 
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Report an error.
        
        Args:
            error: Error to report
            context: Error context
        """
        raise NotImplementedError("Subclasses must implement report method")


class ConsoleReporter(ErrorReporter):
    """Reporter that outputs errors to the console."""
    
    def __init__(
        self, 
        output: TextIO = sys.stderr,
        show_traceback: bool = True,
        color: bool = True
    ):
        """
        Initialize the console reporter.
        
        Args:
            output: Output stream
            show_traceback: Whether to show the traceback
            color: Whether to use color in the output
        """
        self.output = output
        self.show_traceback = show_traceback
        self.color = color
        
    def report(
        self, 
        error: PaperRevisionError, 
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Report an error to the console.
        
        Args:
            error: Error to report
            context: Error context
        """
        context = context or {}
        
        # Get error info
        error_info = error.get_error_info()
        
        # Determine colors
        if self.color:
            red = "\033[31m"
            yellow = "\033[33m"
            green = "\033[32m"
            blue = "\033[34m"
            bold = "\033[1m"
            reset = "\033[0m"
        else:
            red = yellow = green = blue = bold = reset = ""
        
        # Print error header
        self.output.write(f"\n{bold}{red}ERROR: {error_info['error_type']}{reset}\n")
        self.output.write(f"{bold}Message:{reset} {error_info['message']}\n")
        
        # Print context if available
        if context:
            self.output.write(f"\n{bold}Context:{reset}\n")
            if "function_name" in context:
                self.output.write(f"  Function: {context['function_name']}\n")
            
            # Print other context items
            for key, value in context.items():
                if key != "function_name" and key != "args" and key != "kwargs":
                    self.output.write(f"  {key}: {value}\n")
        
        # Print error details
        if error_info["details"]:
            self.output.write(f"\n{bold}Details:{reset}\n")
            for key, value in error_info["details"].items():
                if value is not None:
                    self.output.write(f"  {key}: {value}\n")
        
        # Print cause if available
        if error_info["cause"]:
            self.output.write(f"\n{bold}Caused by:{reset} {error_info['cause']['type']}: {error_info['cause']['message']}\n")
        
        # Print traceback if available and requested
        if self.show_traceback and error_info["traceback"]:
            self.output.write(f"\n{bold}Traceback:{reset}\n")
            for line in error_info["traceback"]:
                self.output.write(f"  {line}")
        
        # Print recovery info
        self.output.write(f"\n{bold}Recoverable:{reset} {green if error_info['recoverable'] else red}{error_info['recoverable']}{reset}\n")
        
        self.output.write("\n")
        self.output.flush()


class FileReporter(ErrorReporter):
    """Reporter that outputs errors to a file."""
    
    def __init__(
        self, 
        file_path: str,
        mode: str = "a",
        show_traceback: bool = True,
        include_timestamp: bool = True
    ):
        """
        Initialize the file reporter.
        
        Args:
            file_path: Path to the error log file
            mode: File open mode
            show_traceback: Whether to show the traceback
            include_timestamp: Whether to include a timestamp
        """
        self.file_path = file_path
        self.mode = mode
        self.show_traceback = show_traceback
        self.include_timestamp = include_timestamp
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
    def report(
        self, 
        error: PaperRevisionError, 
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Report an error to a file.
        
        Args:
            error: Error to report
            context: Error context
        """
        context = context or {}
        
        # Get error info
        error_info = error.get_error_info()
        
        # Open the file
        with open(self.file_path, self.mode) as f:
            # Write timestamp if requested
            if self.include_timestamp:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"[{timestamp}] ")
                
            # Write error header
            f.write(f"ERROR: {error_info['error_type']}\n")
            f.write(f"Message: {error_info['message']}\n")
            
            # Write context if available
            if context:
                f.write("\nContext:\n")
                if "function_name" in context:
                    f.write(f"  Function: {context['function_name']}\n")
                
                # Write other context items
                for key, value in context.items():
                    if key != "function_name" and key != "args" and key != "kwargs":
                        f.write(f"  {key}: {value}\n")
            
            # Write error details
            if error_info["details"]:
                f.write("\nDetails:\n")
                for key, value in error_info["details"].items():
                    if value is not None:
                        f.write(f"  {key}: {value}\n")
            
            # Write cause if available
            if error_info["cause"]:
                f.write(f"\nCaused by: {error_info['cause']['type']}: {error_info['cause']['message']}\n")
            
            # Write traceback if available and requested
            if self.show_traceback and error_info["traceback"]:
                f.write("\nTraceback:\n")
                for line in error_info["traceback"]:
                    f.write(f"  {line}")
            
            # Write recovery info
            f.write(f"\nRecoverable: {error_info['recoverable']}\n")
            
            # Write separator
            f.write("\n" + "-" * 80 + "\n\n")


class JsonReporter(ErrorReporter):
    """Reporter that outputs errors as JSON."""
    
    def __init__(
        self, 
        file_path: Optional[str] = None,
        output: Optional[TextIO] = None,
        include_timestamp: bool = True
    ):
        """
        Initialize the JSON reporter.
        
        Args:
            file_path: Path to the error log file, or None to use output
            output: Output stream, or None to use file_path
            include_timestamp: Whether to include a timestamp
        """
        if file_path is None and output is None:
            raise ValueError("Either file_path or output must be provided")
            
        self.file_path = file_path
        self.output = output
        self.include_timestamp = include_timestamp
        
        # Ensure directory exists if file_path is provided
        if file_path:
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
    def report(
        self, 
        error: PaperRevisionError, 
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Report an error as JSON.
        
        Args:
            error: Error to report
            context: Error context
        """
        context = context or {}
        
        # Get error info
        error_info = error.get_error_info()
        
        # Create report data
        report = {
            "error": error_info,
            "context": context
        }
        
        # Add timestamp if requested
        if self.include_timestamp:
            report["timestamp"] = datetime.now().isoformat()
        
        # Convert to JSON
        json_data = json.dumps(report, indent=2)
        
        # Write to file or output
        if self.file_path:
            with open(self.file_path, "a") as f:
                f.write(json_data)
                f.write("\n")
        else:
            self.output.write(json_data)
            self.output.write("\n")
            self.output.flush()


class LoggingReporter(ErrorReporter):
    """Reporter that outputs errors to a logger."""
    
    def __init__(
        self, 
        logger: Optional[logging.Logger] = None,
        level: int = logging.ERROR
    ):
        """
        Initialize the logging reporter.
        
        Args:
            logger: Logger to use, or None to use the root logger
            level: Logging level
        """
        self.logger = logger or logging.getLogger()
        self.level = level
        
    def report(
        self, 
        error: PaperRevisionError, 
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Report an error to a logger.
        
        Args:
            error: Error to report
            context: Error context
        """
        context = context or {}
        
        # Get error info
        error_info = error.get_error_info()
        
        # Create message
        message = f"{error_info['error_type']}: {error_info['message']}"
        
        # Log with context
        self.logger.log(
            self.level,
            message,
            extra={"error_info": error_info, "context": context}
        )


class CompositeReporter(ErrorReporter):
    """Reporter that delegates to multiple reporters."""
    
    def __init__(self, reporters: List[ErrorReporter]):
        """
        Initialize the composite reporter.
        
        Args:
            reporters: List of reporters to delegate to
        """
        self.reporters = reporters
        
    def report(
        self, 
        error: PaperRevisionError, 
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Report an error to all reporters.
        
        Args:
            error: Error to report
            context: Error context
        """
        for reporter in self.reporters:
            try:
                reporter.report(error, context)
            except Exception as e:
                # Log error but continue
                logging.error(f"Error in reporter {reporter.__class__.__name__}: {str(e)}")


class CallbackReporter(ErrorReporter):
    """Reporter that calls a callback function."""
    
    def __init__(
        self, 
        callback: Callable[[PaperRevisionError, Dict[str, Any]], None]
    ):
        """
        Initialize the callback reporter.
        
        Args:
            callback: Callback function to call
        """
        self.callback = callback
        
    def report(
        self, 
        error: PaperRevisionError, 
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Report an error to the callback.
        
        Args:
            error: Error to report
            context: Error context
        """
        self.callback(error, context or {})


# Global reporter instance
_reporter = None


def get_reporter() -> ErrorReporter:
    """
    Get the global error reporter instance.
    
    Returns:
        Global error reporter
    """
    global _reporter
    if _reporter is None:
        _reporter = ConsoleReporter()
    return _reporter


def set_reporter(reporter: ErrorReporter) -> None:
    """
    Set the global error reporter instance.
    
    Args:
        reporter: Error reporter to use
    """
    global _reporter
    _reporter = reporter


def report_error(
    error: PaperRevisionError, 
    context: Optional[Dict[str, Any]] = None
) -> None:
    """
    Report an error using the global reporter.
    
    Args:
        error: Error to report
        context: Error context
    """
    get_reporter().report(error, context)


def create_default_reporter(
    log_dir: Optional[str] = None,
    console: bool = True,
    file: bool = True,
    json: bool = False,
    logger: Optional[logging.Logger] = None
) -> ErrorReporter:
    """
    Create a default reporter configuration.
    
    Args:
        log_dir: Directory for log files, or None to use current directory
        console: Whether to include console reporting
        file: Whether to include file reporting
        json: Whether to include JSON reporting
        logger: Logger to use for logging reporting
        
    Returns:
        Configured error reporter
    """
    reporters = []
    
    # Set up log directory
    if log_dir is None and (file or json):
        log_dir = os.getcwd()
    
    # Add console reporter
    if console:
        reporters.append(ConsoleReporter())
    
    # Add file reporter
    if file and log_dir:
        file_path = os.path.join(log_dir, "errors.log")
        reporters.append(FileReporter(file_path))
    
    # Add JSON reporter
    if json and log_dir:
        file_path = os.path.join(log_dir, "errors.json")
        reporters.append(JsonReporter(file_path))
    
    # Add logging reporter
    if logger:
        reporters.append(LoggingReporter(logger))
    
    # Create composite reporter
    return CompositeReporter(reporters)