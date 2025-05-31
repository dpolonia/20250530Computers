"""
Performance monitoring utilities for the Paper Revision System.

This module provides tools for monitoring performance of various components,
including:
- Function execution timing
- Memory usage tracking
- Resource consumption metrics
- Performance profiling
- Bottleneck identification
"""

import os
import sys
import time
import json
import psutil
import logging
import functools
import threading
import tracemalloc
from typing import Dict, Any, Optional, List, Callable, Union
from datetime import datetime
from contextlib import contextmanager

# Configure logging
logger = logging.getLogger(__name__)

# Global performance tracking
_performance_metrics = {}
_performance_timers = {}
_lock = threading.RLock()  # Thread-safe access to metrics


class PerformanceMetric:
    """Represents a performance metric with statistics."""
    
    def __init__(self, name: str, category: str = "general"):
        """Initialize a performance metric.
        
        Args:
            name: Name of the metric
            category: Category of the metric
        """
        self.name = name
        self.category = category
        self.count = 0
        self.total_time = 0.0
        self.min_time = float('inf')
        self.max_time = 0.0
        self.avg_time = 0.0
        self.last_time = 0.0
        self.timestamps = []
        self.memory_used = 0
        self.metadata = {}
        self.start_time = None
    
    def start(self):
        """Start timing this metric."""
        self.start_time = time.time()
        return self
    
    def stop(self, memory_used: int = 0, **metadata):
        """Stop timing and record statistics.
        
        Args:
            memory_used: Memory used during operation in bytes
            **metadata: Additional metadata for this measurement
        
        Returns:
            Elapsed time in milliseconds
        """
        if self.start_time is None:
            logger.warning(f"Attempted to stop metric {self.name} that wasn't started")
            return 0
            
        elapsed = (time.time() - self.start_time) * 1000  # Convert to ms
        self.count += 1
        self.total_time += elapsed
        self.min_time = min(self.min_time, elapsed)
        self.max_time = max(self.max_time, elapsed)
        self.avg_time = self.total_time / self.count
        self.last_time = elapsed
        self.timestamps.append(datetime.now())
        
        # Trim timestamps to keep only the most recent 100
        if len(self.timestamps) > 100:
            self.timestamps = self.timestamps[-100:]
        
        # Record memory usage
        if memory_used > 0:
            self.memory_used = memory_used
        
        # Record metadata
        self.metadata.update(metadata)
        
        self.start_time = None
        return elapsed
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.
        
        Returns:
            Dictionary representation of this metric
        """
        return {
            'name': self.name,
            'category': self.category,
            'count': self.count,
            'total_time_ms': self.total_time,
            'min_time_ms': self.min_time if self.min_time != float('inf') else 0,
            'max_time_ms': self.max_time,
            'avg_time_ms': self.avg_time,
            'last_time_ms': self.last_time,
            'last_timestamp': self.timestamps[-1].isoformat() if self.timestamps else None,
            'memory_used_bytes': self.memory_used,
            'metadata': self.metadata
        }


def get_metric(name: str, category: str = "general") -> PerformanceMetric:
    """Get or create a performance metric.
    
    Args:
        name: Name of the metric
        category: Category of the metric
        
    Returns:
        PerformanceMetric instance
    """
    key = f"{category}:{name}"
    with _lock:
        if key not in _performance_metrics:
            _performance_metrics[key] = PerformanceMetric(name, category)
        return _performance_metrics[key]


@contextmanager
def measure_performance(name: str, category: str = "general", log_level: int = logging.DEBUG, 
                      trace_memory: bool = False, log_on_exit: bool = True, 
                      additional_metadata: Dict[str, Any] = None):
    """Context manager for measuring performance of a code block.
    
    Args:
        name: Name of the metric
        category: Category of the metric
        log_level: Logging level for performance logs
        trace_memory: Whether to trace memory usage
        log_on_exit: Whether to log metrics when exiting context
        additional_metadata: Additional metadata to include
        
    Yields:
        PerformanceMetric instance
    """
    metric = get_metric(name, category)
    metadata = additional_metadata or {}
    
    # Start memory tracing if requested
    if trace_memory:
        tracemalloc.start()
        _, start_peak = tracemalloc.get_traced_memory()
    
    try:
        # Start timing
        metric.start()
        
        # Yield control to the context body
        yield metric
        
    finally:
        # Calculate memory usage if tracing
        memory_used = 0
        if trace_memory:
            _, end_peak = tracemalloc.get_traced_memory()
            memory_used = end_peak - start_peak
            tracemalloc.stop()
        
        # Stop timing and record metrics
        elapsed = metric.stop(memory_used=memory_used, **metadata)
        
        # Log metrics if requested
        if log_on_exit:
            logger.log(log_level, f"PERFORMANCE: {category}:{name} - "
                                f"{elapsed:.2f}ms - Count: {metric.count} - "
                                f"Avg: {metric.avg_time:.2f}ms - "
                                f"Memory: {memory_used/1024:.2f}KB")


def measure_func(name: str = None, category: str = "function", log_level: int = logging.DEBUG,
               trace_memory: bool = False, log_on_exit: bool = True):
    """Decorator for measuring function performance.
    
    Args:
        name: Name of the metric (defaults to function name)
        category: Category of the metric
        log_level: Logging level for performance logs
        trace_memory: Whether to trace memory usage
        log_on_exit: Whether to log metrics when exiting
        
    Returns:
        Decorated function
    """
    def decorator(func):
        metric_name = name or func.__name__
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with measure_performance(
                metric_name, category, log_level, trace_memory, log_on_exit,
                {'args': str(args), 'kwargs': str(kwargs)}
            ):
                return func(*args, **kwargs)
                
        return wrapper
        
    return decorator


def get_system_resources() -> Dict[str, Any]:
    """Get current system resource usage.
    
    Returns:
        Dictionary with system resource information
    """
    process = psutil.Process(os.getpid())
    
    # Calculate CPU usage (percentage)
    cpu_percent = process.cpu_percent(interval=0.1)
    
    # Calculate memory usage
    memory_info = process.memory_info()
    
    # Get other metrics
    io_counters = process.io_counters()
    
    return {
        'timestamp': datetime.now().isoformat(),
        'cpu_percent': cpu_percent,
        'memory_rss_bytes': memory_info.rss,
        'memory_vms_bytes': memory_info.vms,
        'memory_percent': process.memory_percent(),
        'threads_count': process.num_threads(),
        'io_read_bytes': io_counters.read_bytes,
        'io_write_bytes': io_counters.write_bytes,
        'io_read_count': io_counters.read_count,
        'io_write_count': io_counters.write_count
    }


def log_system_resources(log_level: int = logging.DEBUG):
    """Log current system resource usage.
    
    Args:
        log_level: Logging level for resource logs
    """
    resources = get_system_resources()
    logger.log(log_level, f"RESOURCES: "
              f"CPU: {resources['cpu_percent']:.1f}% - "
              f"Memory: {resources['memory_rss_bytes']/1024/1024:.1f}MB "
              f"({resources['memory_percent']:.1f}%) - "
              f"Threads: {resources['threads_count']}")
    return resources


def monitor_resources(interval: float = 10.0, callback: Callable = None, 
                    log_level: int = logging.DEBUG):
    """Start a background thread to monitor system resources.
    
    Args:
        interval: Interval between measurements in seconds
        callback: Optional callback function to call with resources
        log_level: Logging level for resource logs
        
    Returns:
        The monitoring thread (call .stop() to stop monitoring)
    """
    stop_event = threading.Event()
    
    def monitor_thread():
        while not stop_event.is_set():
            resources = log_system_resources(log_level)
            if callback:
                callback(resources)
            stop_event.wait(interval)
    
    thread = threading.Thread(target=monitor_thread, daemon=True)
    thread.start()
    
    # Add stop method to thread
    thread.stop = lambda: stop_event.set()
    
    return thread


def get_all_metrics() -> Dict[str, Dict[str, Any]]:
    """Get all performance metrics.
    
    Returns:
        Dictionary mapping metric keys to metric dictionaries
    """
    with _lock:
        return {key: metric.to_dict() for key, metric in _performance_metrics.items()}


def reset_metrics():
    """Reset all performance metrics."""
    with _lock:
        _performance_metrics.clear()


def export_metrics(filepath: str):
    """Export metrics to a file.
    
    Args:
        filepath: Path to the file to export metrics to
    """
    metrics = get_all_metrics()
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Exported {len(metrics)} performance metrics to {filepath}")


def generate_performance_report() -> str:
    """Generate a text-based performance report.
    
    Returns:
        Performance report as a string
    """
    metrics = get_all_metrics()
    
    # Group metrics by category
    categories = {}
    for key, metric in metrics.items():
        category = metric['category']
        if category not in categories:
            categories[category] = []
        categories[category].append(metric)
    
    # Build the report
    report = ["# Performance Report", ""]
    report.append(f"Generated: {datetime.now().isoformat()}")
    report.append(f"Total Metrics: {len(metrics)}")
    report.append("")
    
    # System resources
    resources = get_system_resources()
    report.append("## System Resources")
    report.append("")
    report.append(f"- CPU Usage: {resources['cpu_percent']:.1f}%")
    report.append(f"- Memory Usage: {resources['memory_rss_bytes']/1024/1024:.1f}MB ({resources['memory_percent']:.1f}%)")
    report.append(f"- Threads: {resources['threads_count']}")
    report.append("")
    
    # Metrics by category
    for category, category_metrics in sorted(categories.items()):
        report.append(f"## {category.title()}")
        report.append("")
        
        # Sort metrics by total time
        category_metrics.sort(key=lambda x: x['total_time_ms'], reverse=True)
        
        # Table header
        report.append("| Metric | Count | Total Time | Avg Time | Min Time | Max Time | Last Time |")
        report.append("|--------|-------|------------|----------|----------|----------|-----------|")
        
        # Table rows
        for metric in category_metrics:
            report.append(
                f"| {metric['name']} | {metric['count']} | "
                f"{metric['total_time_ms']:.1f}ms | {metric['avg_time_ms']:.1f}ms | "
                f"{metric['min_time_ms']:.1f}ms | {metric['max_time_ms']:.1f}ms | "
                f"{metric['last_time_ms']:.1f}ms |"
            )
        
        report.append("")
    
    return "\n".join(report)


def find_bottlenecks(threshold: float = 100.0) -> List[Dict[str, Any]]:
    """Find performance bottlenecks.
    
    Args:
        threshold: Threshold in milliseconds for bottleneck detection
        
    Returns:
        List of bottleneck metrics
    """
    metrics = get_all_metrics()
    bottlenecks = []
    
    for key, metric in metrics.items():
        # Check if average time exceeds threshold
        if metric['avg_time_ms'] >= threshold:
            bottlenecks.append(metric)
    
    # Sort bottlenecks by average time
    bottlenecks.sort(key=lambda x: x['avg_time_ms'], reverse=True)
    
    return bottlenecks


def log_bottlenecks(threshold: float = 100.0, log_level: int = logging.WARNING):
    """Log performance bottlenecks.
    
    Args:
        threshold: Threshold in milliseconds for bottleneck detection
        log_level: Logging level for bottleneck logs
    """
    bottlenecks = find_bottlenecks(threshold)
    
    if bottlenecks:
        logger.log(log_level, f"Found {len(bottlenecks)} performance bottlenecks:")
        for bottleneck in bottlenecks:
            logger.log(log_level, f"BOTTLENECK: {bottleneck['category']}:{bottleneck['name']} - "
                                f"Avg: {bottleneck['avg_time_ms']:.1f}ms - "
                                f"Count: {bottleneck['count']}")
    
    return bottlenecks


def profile_function(func: Callable, *args, **kwargs) -> Dict[str, Any]:
    """Profile a function execution.
    
    Args:
        func: Function to profile
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        Profiling results
    """
    import cProfile
    import pstats
    import io
    
    # Start profiling
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run the function
    result = func(*args, **kwargs)
    
    # Stop profiling
    profiler.disable()
    
    # Get statistics
    s = io.StringIO()
    stats = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    stats.print_stats(20)  # Print top 20 functions
    
    # Format the results
    profiling_results = {
        'function': func.__name__,
        'args': str(args),
        'kwargs': str(kwargs),
        'stats': s.getvalue()
    }
    
    return profiling_results