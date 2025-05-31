"""
Logging configuration for the Paper Revision System.

This module provides a centralized logging configuration system with support for:
- Multiple log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- File and console logging
- Log rotation
- Contextual logging with correlation IDs
- Sensitive data masking
- Performance metrics logging
"""

import os
import json
import uuid
import logging
import logging.config
import logging.handlers
import threading
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime

# Constants
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_FORMAT = "%(asctime)s [%(levelname)s] [%(correlation_id)s] %(name)s: %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs")

# Make sure the log directory exists
os.makedirs(DEFAULT_LOG_DIR, exist_ok=True)

# Thread local storage for correlation IDs
local_storage = threading.local()


class CorrelationIdFilter(logging.Filter):
    """Filter that adds correlation ID to log records."""
    
    def filter(self, record):
        """Add correlation_id attribute to log record."""
        if not hasattr(record, 'correlation_id'):
            record.correlation_id = getattr(local_storage, 'correlation_id', '-')
        return True


class SensitiveDataFilter(logging.Filter):
    """Filter that masks sensitive data in log messages."""
    
    def __init__(self, patterns: List[str] = None):
        """Initialize the filter with patterns to mask.
        
        Args:
            patterns: List of patterns to mask (e.g., 'api_key', 'password')
        """
        super().__init__()
        self.patterns = patterns or ['api_key', 'key', 'password', 'secret', 'token', 'credential']
    
    def filter(self, record):
        """Mask sensitive data in log message."""
        if record.getMessage():
            msg = record.getMessage()
            
            # Check if any sensitive patterns are in the message
            for pattern in self.patterns:
                # Look for pattern="value" or pattern='value' or pattern:value
                # or "pattern": "value" or 'pattern': 'value'
                for prefix in [f'{pattern}=', f'{pattern}:', f'"{pattern}":', f"'{pattern}':"]:
                    if prefix in msg.lower():
                        # Find the value after the pattern
                        start_idx = msg.lower().find(prefix) + len(prefix)
                        
                        # Skip whitespace
                        while start_idx < len(msg) and msg[start_idx].isspace():
                            start_idx += 1
                        
                        # Determine if value is quoted
                        quote_char = None
                        if start_idx < len(msg) and msg[start_idx] in ['"', "'"]:
                            quote_char = msg[start_idx]
                            start_idx += 1
                        
                        # Find the end of the value
                        if quote_char:
                            end_idx = msg.find(quote_char, start_idx)
                            if end_idx == -1:
                                end_idx = len(msg)
                        else:
                            # Find the next whitespace or comma or closing brace/bracket
                            for end_idx in range(start_idx, len(msg)):
                                if msg[end_idx].isspace() or msg[end_idx] in [',', '}', ']']:
                                    break
                            if end_idx == len(msg) - 1:
                                end_idx = len(msg)
                        
                        # Mask the value
                        value = msg[start_idx:end_idx]
                        if len(value) > 4:
                            # Mask as: first 2 chars + ****** + last 2 chars
                            masked_value = value[:2] + '*' * 6 + value[-2:]
                            msg = msg[:start_idx] + masked_value + msg[end_idx:]
                            record.args = tuple(masked_value if arg == value else arg 
                                             for arg in record.args) if record.args else ()
            
            # Set the modified message back to the record
            record.msg = msg
            
        return True


class PerformanceMetricsFilter(logging.Filter):
    """Filter that adds performance metrics to log records."""
    
    def __init__(self):
        """Initialize the filter."""
        super().__init__()
        self.start_times = {}
    
    def start_timer(self, name: str):
        """Start a timer for a named operation.
        
        Args:
            name: Name of the operation
        """
        self.start_times[name] = datetime.now()
    
    def stop_timer(self, name: str) -> Optional[float]:
        """Stop a timer and return the elapsed time.
        
        Args:
            name: Name of the operation
            
        Returns:
            Elapsed time in milliseconds, or None if timer not found
        """
        if name in self.start_times:
            elapsed = (datetime.now() - self.start_times[name]).total_seconds() * 1000
            del self.start_times[name]
            return elapsed
        return None
    
    def filter(self, record):
        """Add performance metrics to log record."""
        # Add a method to create performance log message
        def performance_log(name, metrics):
            record.msg = f"PERFORMANCE: {name} - {json.dumps(metrics)}"
            return True
            
        record.performance_log = performance_log
        return True


def get_correlation_id() -> str:
    """Get the current correlation ID or create a new one.
    
    Returns:
        Current correlation ID
    """
    if not hasattr(local_storage, 'correlation_id'):
        local_storage.correlation_id = str(uuid.uuid4())
    return local_storage.correlation_id


def set_correlation_id(correlation_id: str = None):
    """Set the correlation ID for the current thread.
    
    Args:
        correlation_id: Correlation ID to set, or None to generate a new one
    """
    local_storage.correlation_id = correlation_id or str(uuid.uuid4())


def clear_correlation_id():
    """Clear the correlation ID for the current thread."""
    if hasattr(local_storage, 'correlation_id'):
        delattr(local_storage, 'correlation_id')


def configure_logging(config_file: str = None, log_level: str = None, 
                    log_to_console: bool = True, log_to_file: bool = True, 
                    log_dir: str = None, log_file: str = None, 
                    max_bytes: int = 10485760, backup_count: int = 5,
                    correlation_enabled: bool = True,
                    mask_sensitive_data: bool = True,
                    performance_metrics: bool = True):
    """Configure logging for the application.
    
    Args:
        config_file: Path to a logging config file (overrides other parameters if provided)
        log_level: Log level to use (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_console: Whether to log to console
        log_to_file: Whether to log to file
        log_dir: Directory to store log files
        log_file: Name of the log file
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup log files to keep
        correlation_enabled: Whether to enable correlation ID logging
        mask_sensitive_data: Whether to mask sensitive data in logs
        performance_metrics: Whether to enable performance metrics logging
    """
    # If a config file is provided, use it
    if config_file and os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
        logging.config.dictConfig(config)
        return
    
    # Default configuration
    log_level = log_level or DEFAULT_LOG_LEVEL
    log_dir = log_dir or DEFAULT_LOG_DIR
    log_file = log_file or "paper_revision.log"
    
    # Create the log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Basic configuration
    handlers = {}
    filters = {}
    
    # Add correlation ID filter
    if correlation_enabled:
        filters['correlation_id_filter'] = {
            '()': CorrelationIdFilter
        }
    
    # Add sensitive data filter
    if mask_sensitive_data:
        filters['sensitive_data_filter'] = {
            '()': SensitiveDataFilter
        }
    
    # Add performance metrics filter
    if performance_metrics:
        filters['performance_metrics_filter'] = {
            '()': PerformanceMetricsFilter
        }
    
    # Console handler
    if log_to_console:
        handlers['console'] = {
            'class': 'logging.StreamHandler',
            'level': log_level,
            'formatter': 'standard',
            'stream': 'ext://sys.stdout'
        }
        
        # Add filters to console handler
        if correlation_enabled or mask_sensitive_data or performance_metrics:
            handlers['console']['filters'] = []
            if correlation_enabled:
                handlers['console']['filters'].append('correlation_id_filter')
            if mask_sensitive_data:
                handlers['console']['filters'].append('sensitive_data_filter')
            if performance_metrics:
                handlers['console']['filters'].append('performance_metrics_filter')
    
    # File handler
    if log_to_file:
        log_file_path = os.path.join(log_dir, log_file)
        handlers['file'] = {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': log_level,
            'formatter': 'standard',
            'filename': log_file_path,
            'maxBytes': max_bytes,
            'backupCount': backup_count,
            'encoding': 'utf8'
        }
        
        # Add filters to file handler
        if correlation_enabled or mask_sensitive_data or performance_metrics:
            handlers['file']['filters'] = []
            if correlation_enabled:
                handlers['file']['filters'].append('correlation_id_filter')
            if mask_sensitive_data:
                handlers['file']['filters'].append('sensitive_data_filter')
            if performance_metrics:
                handlers['file']['filters'].append('performance_metrics_filter')
    
    # Create the logging configuration
    config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': DEFAULT_LOG_FORMAT,
                'datefmt': DEFAULT_DATE_FORMAT
            }
        },
        'filters': filters,
        'handlers': handlers,
        'loggers': {
            '': {
                'handlers': list(handlers.keys()),
                'level': log_level,
                'propagate': True
            }
        }
    }
    
    # Apply the configuration
    logging.config.dictConfig(config)
    logger = logging.getLogger(__name__)
    logger.debug("Logging configured")


# Performance monitoring utilities
performance_metrics = {}
performance_timers = {}

def start_timer(name: str):
    """Start a timer for a named operation.
    
    Args:
        name: Name of the operation
    """
    performance_timers[name] = datetime.now()


def stop_timer(name: str) -> Optional[float]:
    """Stop a timer and return the elapsed time.
    
    Args:
        name: Name of the operation
        
    Returns:
        Elapsed time in milliseconds, or None if timer not found
    """
    if name in performance_timers:
        elapsed = (datetime.now() - performance_timers[name]).total_seconds() * 1000
        del performance_timers[name]
        
        # Store the metric
        if name not in performance_metrics:
            performance_metrics[name] = {
                'count': 0,
                'total_time': 0,
                'min_time': float('inf'),
                'max_time': 0,
                'avg_time': 0
            }
        
        metrics = performance_metrics[name]
        metrics['count'] += 1
        metrics['total_time'] += elapsed
        metrics['min_time'] = min(metrics['min_time'], elapsed)
        metrics['max_time'] = max(metrics['max_time'], elapsed)
        metrics['avg_time'] = metrics['total_time'] / metrics['count']
        
        return elapsed
    return None


def log_performance(logger, name: str, additional_metrics: Dict[str, Any] = None):
    """Log performance metrics for a named operation.
    
    Args:
        logger: Logger to use
        name: Name of the operation
        additional_metrics: Additional metrics to include in the log
    """
    if name in performance_metrics:
        metrics = performance_metrics[name].copy()
        
        # Add additional metrics
        if additional_metrics:
            metrics.update(additional_metrics)
        
        # Log the metrics
        logger.info(f"PERFORMANCE: {name} - {json.dumps(metrics)}")


def get_all_performance_metrics() -> Dict[str, Dict[str, Any]]:
    """Get all performance metrics.
    
    Returns:
        Dictionary of all performance metrics
    """
    return performance_metrics


def reset_performance_metrics():
    """Reset all performance metrics."""
    performance_metrics.clear()
    performance_timers.clear()


# Initialize logging with default configuration
configure_logging()