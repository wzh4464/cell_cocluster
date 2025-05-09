###
# File: src/data/reader.py
# Created Date: Wednesday, March 26th 2025
# Author: Zihan
# -----
# Last Modified: Wednesday, 23rd April 2025 10:46:53 pm
# Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
# -----
# HISTORY:
# Date      		By   	Comments
# ----------		------	---------------------------------------------------------
###

"""
Data reader module for loading and processing NIfTI files.
"""

import os
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Union, List, Dict, Any

def load_nifti_file(file_path: Union[str, Path]) -> np.ndarray:
    """
    Load a NIfTI file and return its data as a numpy array.
    
    Args:
        file_path: Path to the NIfTI file
        
    Returns:
        numpy.ndarray: The volume data
        
    Raises:
        FileNotFoundError: If the file does not exist
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return nib.load(file_path).get_fdata()

def save_nifti_file(data: np.ndarray, file_path: Union[str, Path]) -> None:
    """
    Save a numpy array as a NIfTI file.
    
    Args:
        data: The volume data to save
        file_path: Path where to save the NIfTI file
    """
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(file_path))

def get_all_nifti_files(data_dir: Union[str, Path]) -> List[Path]:
    """
    Get all NIfTI files in the specified directory.
    
    Args:
        data_dir: Directory containing NIfTI files
        
    Returns:
        List[Path]: Sorted list of NIfTI file paths
    """
    data_path = Path(data_dir)
    return sorted(data_path.glob("*_segCell.nii.gz"))

def process_nifti_files(data_dir: Union[str, Path]) -> Dict[str, np.ndarray]:
    """
    Process all NIfTI files in the directory.
    
    Args:
        data_dir: Directory containing NIfTI files
        
    Returns:
        Dict[str, np.ndarray]: Dictionary mapping frame numbers to volume data
    """
    nifti_files = get_all_nifti_files(data_dir)
    
    # Dictionary to store all volumes
    volumes = {}
    
    # Create output directory for npy files if it doesn't exist
    output_dir = Path("data_npy")
    output_dir.mkdir(exist_ok=True)
    
    for file_path in nifti_files:
        try:
            # Extract the frame number from filename (e.g., "001" from "WT_Sample1_001_segCell.nii.gz")
            frame_num = file_path.stem.split('_')[-2]
            
            # Load the volume
            volume = load_nifti_file(str(file_path))
            
            # Store in dictionary with frame number as key
            volumes[frame_num] = volume
            
            # Save as npy file
            npy_path = output_dir / f"frame_{frame_num}.npy"
            np.save(npy_path, volume)
            
            print(f"Successfully loaded and saved frame {frame_num}")
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    return volumes

def main():
    if volumes := process_nifti_files(
        "DATA/SegmentCellUnified/WT_Sample1LabelUnified"
    ):
        first_vol = list(volumes.values())[0]
        print(f"\nData Summary:")
        print(f"Number of frames loaded: {len(volumes)}")
        print(f"Volume shape: {first_vol.shape}")
        print(f"Data type: {first_vol.dtype}")
        print(f"Value range: [{first_vol.min()}, {first_vol.max()}]")
        print(f"\nNPY files have been saved in the 'data_npy' directory")

if __name__ == "__main__":
    main()
