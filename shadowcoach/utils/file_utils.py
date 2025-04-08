"""
File utilities for the ShadowCoach system
"""
import os
import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import pickle

def generate_unique_filename(base_path: str, prefix: str = "", suffix: str = "") -> str:
    """
    Generate a unique filename with timestamp
    
    Args:
        base_path: The base directory path
        prefix: Optional prefix for the filename
        suffix: Optional suffix for the filename
        
    Returns:
        A unique filename with timestamp
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
    Save data to a pickle file
    
    Args:
        data: The data to save
        file_path: The path to save the data to
    """
    # Ensure the directory exists
    ensure_directory_exists(os.path.dirname(file_path))
    
    # Save the data
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

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
