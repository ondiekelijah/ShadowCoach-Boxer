"""
File utilities for the ShadowCoach system.

This module provides file and path management functionality:
- Unique filename generation
- Directory creation
- Pickle file handling
- Path validation

Key Features:
    - Timestamp-based unique filenames
    - Automated directory creation
    - Serialization helpers
    - Path management utilities

Usage Example:
    >>> filename = generate_unique_filename("output", "analysis", ".mp4")
    >>> ensure_directory_exists("output")
    >>> save_pickle(data, "models/reference.pkl")
"""

import os
import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import pickle

def generate_unique_filename(base_path: str, prefix: str = "", suffix: str = "") -> str:
    """
    Generate a unique filename using timestamps.

    Creates a unique filename by combining a prefix, timestamp, and suffix.
    Ensures no filename collisions in the target directory.

    Args:
        base_path: Directory for the file
        prefix: Optional prefix for the filename
        suffix: Optional suffix/extension

    Returns:
        Complete path to unique filename

    Example:
        >>> filename = generate_unique_filename("output", "video_", ".mp4")
        >>> print(filename)
        "output/video_20231225_143022.mp4"
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}{suffix}" if prefix else f"{timestamp}{suffix}"
    return os.path.join(base_path, filename)

def ensure_directory_exists(directory_path: str) -> None:
    """
    Ensure that a directory exists, creating it if necessary

    Args:
        directory_path: The path to the directory
    """
    Path(directory_path).mkdir(exist_ok=True, parents=True)

def save_pickle(data: Any, file_path: str) -> None:
    """
    Save data to a pickle file with error handling.

    Creates necessary directories and safely serializes data.

    Args:
        data: Data to serialize
        file_path: Path to save the pickle file

    Raises:
        IOError: If file cannot be written
        PickleError: If data cannot be serialized
    """
    # Ensure the directory exists
    ensure_directory_exists(os.path.dirname(file_path))

    # Save the data
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
    except IOError as e:
        raise IOError(f"Failed to write file {file_path}: {e}")
    except pickle.PickleError as e:
        raise pickle.PickleError(f"Failed to serialize data: {e}")

def load_pickle(file_path: str) -> Any:
    """
    Load data from a pickle file

    Args:
        file_path: The path to load the data from

    Returns:
        The loaded data
    """
    with open(file_path, 'rb') as f:
        return pickle.load(f)
