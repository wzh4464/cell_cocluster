"""
Utility functions for the cell co-clustering analysis toolkit.
"""

import numpy as np
from pathlib import Path

def ensure_dir(directory):
    """Ensure that a directory exists, create it if it doesn't."""
    Path(directory).mkdir(parents=True, exist_ok=True)

def save_array(array, path):
    """Save a numpy array to disk."""
    np.save(path, array)

def load_array(path):
    """Load a numpy array from disk."""
    return np.load(path)

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points."""
    return np.linalg.norm(np.array(point1) - np.array(point2)) 