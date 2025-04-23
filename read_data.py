###
# File: ./read_data.py
# Created Date: Wednesday, March 26th 2025
# Author: Zihan
# -----
# Last Modified: Wednesday, 23rd April 2025 12:26:12 pm
# Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
# -----
# HISTORY:
# Date      		By   	Comments
# ----------		------	---------------------------------------------------------
###

"""
Data Summary:
Number of frames loaded: 255
Volume shape: (256, 356, 214)
Data type: float64
Value range: [0.0, 1209.0]
"""

import os
import numpy as np
import nibabel as nib
from pathlib import Path

def load_nifti_file(file_path):
    """Load a NIfTI file and return its data as a numpy array."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return nib.load(file_path).get_fdata()

def get_all_nifti_files(data_dir):
    """Get all NIfTI files in the specified directory."""
    data_path = Path(data_dir)
    return sorted(data_path.glob("WT_Sample1_*_segCell.nii.gz"))

def process_nifti_files(data_dir):
    """Process all NIfTI files in the directory."""
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
    # Directory containing the NIfTI files
    data_dir = "data"
    
    # Process all files
    volumes = process_nifti_files(data_dir)
    
    # Print some basic information about the loaded data
    if volumes:
        first_vol = list(volumes.values())[0]
        print(f"\nData Summary:")
        print(f"Number of frames loaded: {len(volumes)}")
        print(f"Volume shape: {first_vol.shape}")
        print(f"Data type: {first_vol.dtype}")
        print(f"Value range: [{first_vol.min()}, {first_vol.max()}]")
        print(f"\nNPY files have been saved in the 'data_npy' directory")

if __name__ == "__main__":
    main()
