import os
import numpy as np
import multiprocessing as mp
from functools import partial
import glob
import pandas as pd
from tqdm import tqdm

# Import 3DCSQ functions
import threeDCSQ.utils.cell_func as cell_f
from threeDCSQ.transformation.SH_represention import sample_and_SHc_with_surface
from threeDCSQ.transformation.SH_represention import get_flatten_ldegree_morder

def process_single_cell(args):
    """Process a single cell from a frame.
    
    Args:
        args: tuple containing (npy_file, cell_label, sample_N, lmax, surface_average_num)
        
    Returns:
        tuple: (frame_num, cell_label, coefficients) or None if error
    """
    npy_file, cell_label, sample_N, lmax, surface_average_num = args
    try:
        # Load frame data
        frame_data = np.load(npy_file)
        frame_num = os.path.basename(npy_file).split('_')[1].split('.')[0]
        
        # Get cell surface points
        cell_surface, center = cell_f.nii_get_cell_surface(frame_data, int(cell_label))
        
        # Calculate spherical harmonics coefficients
        sh_coefficient_instance = sample_and_SHc_with_surface(
            surface_points=cell_surface,
            sample_N=sample_N,
            lmax=lmax,
            surface_average_num=surface_average_num
        )
        from threeDCSQ.utils import sh_cooperation
        coeffs = sh_cooperation.flatten_clim(sh_coefficient_instance.coeffs)
        return (frame_num, cell_label, coeffs)
    except Exception as e:
        print(f"Error processing cell {cell_label} in frame {npy_file}: {str(e)}")
        return None

def calculate_sh_from_npy_parallel(data_npy_dir, saving_path_root, sample_N=30, lmax=14, 
                                 name_dictionary_path=None, surface_average_num=3, n_processes=None):
    """Calculate spherical harmonics coefficients from frame data stored in npy files using parallel processing.
    
    Args:
        data_npy_dir (str): Directory containing the npy files with frame data
        saving_path_root (str): Directory to save the results
        sample_N (int): Number of samples for spherical sampling
        lmax (int): Maximum degree for spherical harmonics
        name_dictionary_path (str, optional): Path to name dictionary CSV file
        surface_average_num (int): Number of points to average for surface calculation
        n_processes (int, optional): Number of processes to use. Defaults to CPU count.
    """
    print(f"Processing frame data from {data_npy_dir}...")

    # Get all npy files
    npy_files = sorted(glob.glob(os.path.join(data_npy_dir, "frame_*.npy")))

    if not npy_files:
        raise ValueError(f"No frame_*.npy files found in {data_npy_dir}")

    # Get cell name mapping if dictionary path is provided
    if name_dictionary_path:
        number_cell_affine_table, _ = cell_f.get_cell_name_affine_table(path=name_dictionary_path)
    else:
        number_cell_affine_table = {}

    # Create DataFrame to store coefficients
    column_indices = get_flatten_ldegree_morder(lmax)
    df_coeffs = pd.DataFrame(columns=column_indices)

    # Prepare all cell processing arguments
    all_cell_args = []
    for npy_file in npy_files:
        # Load frame data to get cell labels
        frame_data = np.load(npy_file)
        cell_labels = np.unique(frame_data)
        cell_labels = cell_labels[cell_labels != 0]  # Remove background

        # Add arguments for each cell in this frame
        all_cell_args.extend([
            (npy_file, label, sample_N, lmax, surface_average_num)
            for label in cell_labels
        ])

    # Process all cells in parallel
    if n_processes is None:
        n_processes = mp.cpu_count()

    print(f"Processing {len(all_cell_args)} cells using {n_processes} processes")
    with mp.Pool(n_processes) as pool:
        results = list(tqdm(
            pool.imap(process_single_cell, all_cell_args),
            total=len(all_cell_args),
            desc="Processing cells"
        ))

    # Process results
    for result in results:
        if result is not None:
            frame_num, cell_label, coeffs = result
            # Get cell name from dictionary or use default
            cell_name = number_cell_affine_table.get(int(cell_label), f"cell_{int(cell_label)}")
            # Add to DataFrame
            df_coeffs.loc[f"{frame_num}::{cell_name}"] = coeffs.tolist()

    # Save results
    os.makedirs(saving_path_root, exist_ok=True)
    output_file = os.path.join(saving_path_root, f"sh_coefficients_l{lmax+1}.csv")
    df_coeffs.to_csv(output_file)
    print(f"Results saved to {output_file}")

def main():
    # Directory containing the frame npy files
    data_npy_dir = "data_npy"
    
    # Directory to save the results
    saving_path_root = "DATA/spharm_parallel"
    
    # Calculate spherical harmonics coefficients with parallel processing
    calculate_sh_from_npy_parallel(
        data_npy_dir=data_npy_dir,
        saving_path_root=saving_path_root,
        sample_N=30,
        lmax=14,
        name_dictionary_path='DATA/name_dictionary.csv',
        surface_average_num=3,
        n_processes=None  # Will use all available CPU cores
    )

if __name__ == "__main__":
    main() 
