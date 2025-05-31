"""
Path utilities for the paper revision tool.

This module provides utilities for file path handling, including path construction,
validation, and normalization.
"""

import os
import pathlib
import datetime
import tempfile
import logging
from typing import Optional, List, Union, Tuple


logger = logging.getLogger(__name__)


def ensure_directory_exists(directory_path: str) -> str:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory_path: Path to the directory
        
    Returns:
        The directory path
    """
    os.makedirs(directory_path, exist_ok=True)
    return directory_path


def ensure_parent_directory_exists(file_path: str) -> str:
    """
    Ensure the parent directory of a file exists, creating it if necessary.
    
    Args:
        file_path: Path to the file
        
    Returns:
        The file path
    """
    directory = os.path.dirname(file_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    return file_path


def get_current_timestamp(format_string: str = "%Y%m%d_%H%M%S") -> str:
    """
    Get the current timestamp as a formatted string.
    
    Args:
        format_string: Format string for the timestamp
        
    Returns:
        Formatted timestamp string
    """
    return datetime.datetime.now().strftime(format_string)


def normalize_path(file_path: str) -> str:
    """
    Normalize a file path.
    
    Args:
        file_path: Path to normalize
        
    Returns:
        Normalized path
    """
    return os.path.normpath(os.path.abspath(file_path))


def get_filename(file_path: str, with_extension: bool = True) -> str:
    """
    Get the filename from a path.
    
    Args:
        file_path: Path to the file
        with_extension: Whether to include the extension
        
    Returns:
        Filename
    """
    if with_extension:
        return os.path.basename(file_path)
    return os.path.splitext(os.path.basename(file_path))[0]


def get_extension(file_path: str) -> str:
    """
    Get the extension from a file path.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File extension (with dot)
    """
    return os.path.splitext(file_path)[1]


def is_valid_file(file_path: str, check_readable: bool = True) -> bool:
    """
    Check if a file exists and is valid.
    
    Args:
        file_path: Path to the file
        check_readable: Whether to check if the file is readable
        
    Returns:
        True if the file is valid, False otherwise
    """
    if not os.path.exists(file_path):
        return False
    if not os.path.isfile(file_path):
        return False
    if check_readable and not os.access(file_path, os.R_OK):
        return False
    return True


def is_valid_directory(directory_path: str, check_writable: bool = False) -> bool:
    """
    Check if a directory exists and is valid.
    
    Args:
        directory_path: Path to the directory
        check_writable: Whether to check if the directory is writable
        
    Returns:
        True if the directory is valid, False otherwise
    """
    if not os.path.exists(directory_path):
        return False
    if not os.path.isdir(directory_path):
        return False
    if check_writable and not os.access(directory_path, os.W_OK):
        return False
    return True


def construct_output_path(
    base_name: str,
    output_dir: Optional[str] = None,
    original_file_path: Optional[str] = None,
    extension: str = ".docx",
    include_timestamp: bool = True
) -> str:
    """
    Construct an output file path.
    
    Args:
        base_name: Base name for the output file
        output_dir: Output directory, or None to use the directory of original_file_path
        original_file_path: Original file path, used if output_dir is None
        extension: File extension for the output file
        include_timestamp: Whether to include a timestamp in the filename
        
    Returns:
        Output file path
    """
    # Determine output directory
    if output_dir:
        directory = output_dir
    elif original_file_path:
        directory = os.path.dirname(original_file_path)
    else:
        directory = os.getcwd()
    
    # Ensure directory exists
    ensure_directory_exists(directory)
    
    # Construct filename
    if include_timestamp:
        timestamp = get_current_timestamp()
        filename = f"{base_name}_{timestamp}{extension}"
    else:
        filename = f"{base_name}{extension}"
    
    # Construct full path
    return os.path.join(directory, filename)


def list_files_with_extension(
    directory: str,
    extension: str,
    recursive: bool = False
) -> List[str]:
    """
    List files with a specific extension in a directory.
    
    Args:
        directory: Directory to search
        extension: File extension to filter by (with or without dot)
        recursive: Whether to search recursively
        
    Returns:
        List of file paths
    """
    # Normalize extension to include dot
    if not extension.startswith("."):
        extension = f".{extension}"
    
    result = []
    
    if recursive:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(extension):
                    result.append(os.path.join(root, file))
    else:
        for file in os.listdir(directory):
            if file.endswith(extension):
                result.append(os.path.join(directory, file))
    
    return result


def create_temp_file(content: str = "", suffix: str = ".txt") -> str:
    """
    Create a temporary file with optional content.
    
    Args:
        content: Content to write to the file
        suffix: File extension
        
    Returns:
        Path to the temporary file
    """
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
        if content:
            temp.write(content.encode("utf-8"))
        return temp.name


def create_temp_directory() -> str:
    """
    Create a temporary directory.
    
    Returns:
        Path to the temporary directory
    """
    return tempfile.mkdtemp()


def split_path(file_path: str) -> Tuple[str, str, str]:
    """
    Split a file path into directory, filename, and extension.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Tuple of (directory, filename, extension)
    """
    directory = os.path.dirname(file_path)
    filename_with_ext = os.path.basename(file_path)
    filename, extension = os.path.splitext(filename_with_ext)
    return directory, filename, extension


def join_path(*parts: str) -> str:
    """
    Join path parts.
    
    Args:
        *parts: Path parts to join
        
    Returns:
        Joined path
    """
    return os.path.join(*parts)


def expand_path(path: str) -> str:
    """
    Expand user and environment variables in a path.
    
    Args:
        path: Path to expand
        
    Returns:
        Expanded path
    """
    return os.path.expanduser(os.path.expandvars(path))


def resolve_relative_path(path: str, base_path: Optional[str] = None) -> str:
    """
    Resolve a relative path against a base path.
    
    Args:
        path: Path to resolve
        base_path: Base path to resolve against, or None to use the current directory
        
    Returns:
        Resolved path
    """
    if base_path is None:
        base_path = os.getcwd()
    
    if os.path.isabs(path):
        return path
    
    return os.path.normpath(os.path.join(base_path, path))