###
# File: ./aggregate_features.py
# Created Date: Tuesday, April 29th 2025
# Author: Zihan
# -----
# Last Modified: Tuesday, 29th April 2025 11:23:02 am
# Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
# -----
# HISTORY:
# Date      		By   	Comments
# ----------		------	---------------------------------------------------------
###

import numpy as np
import pandas as pd
from pathlib import Path
import json
from tqdm import tqdm
import re

# --- Configuration ---
GEO_FEATURES_DIR = Path("DATA/geo_features")
SPHARM_DIR = Path("DATA/spharm")
NAME_DICT_PATH = Path("DATA/name_dictionary.csv")
OUTPUT_TENSOR_PATH = Path("DATA/aggregated_features.npy")
OUTPUT_METADATA_PATH = Path("DATA/aggregated_metadata.json")
SH_DEGREE = 15 # Revert SH degree based on actual filenames (l15)

# Features from kinematic_features.py
KINEMATIC_FEATURES = [
    'volume', 'surface_area',
    'centroid_x', 'centroid_y', 'centroid_z',
    'velocity_x', 'velocity_y', 'velocity_z',
    'acceleration_x', 'acceleration_y', 'acceleration_z'
]
N_KINEMATIC = len(KINEMATIC_FEATURES)

# Spherical Harmonic features
N_SH = (SH_DEGREE)**2
SH_FEATURES = [f'sh_{i}' for i in range(N_SH)]

ALL_FEATURES = KINEMATIC_FEATURES + SH_FEATURES
N_FEATURES = len(ALL_FEATURES)

# --- Helper Functions ---
def load_name_dictionary(path):
    """Loads the name dictionary, mapping numeric ID (float) to name."""
    try:
        return load_name_dictionary(path)
    except FileNotFoundError:
        print(f"Error: Name dictionary not found at {path}")
        return None, None, None
    except Exception as e:
        print(f"Error loading name dictionary: {e}")
        return None, None, None


# TODO Rename this here and in `load_name_dictionary`
def load_name_dictionary(path):
    df = pd.read_csv(path, header=None, index_col=0, names=['cell_name'])
    name_to_id_str = {}
    valid_cell_names = []
    for idx, name in df['cell_name'].items():
        # Check if idx is NaN or invalid before converting to int
        if pd.notna(idx):
            try:
                numeric_id_str = f"{int(idx):03d}"
                name_to_id_str[name] = numeric_id_str
                valid_cell_names.append(name)
            except (ValueError, TypeError):
                print(f"Warning: Skipping row with invalid index '{idx}' and name '{name}' in name dictionary.")
        else:
            print(f"Warning: Skipping row with NaN index and name '{name}' in name dictionary.")

    # Create mapping from padded numeric string ID to name
    id_str_to_name = {v: k for k, v in name_to_id_str.items()}
    print(f"Loaded {len(valid_cell_names)} valid entries from name dictionary.")
    return name_to_id_str, id_str_to_name, valid_cell_names

def discover_cells_and_timepoints(geo_dir, spharm_dir, all_cell_names_from_dict):
    """Discovers unique cells and timepoints from feature directories."""
    timepoints = set()
    # Scan geo_features for numeric IDs and timepoints
    if geo_dir.exists():
        numeric_ids_found = set()

        for cell_dir in geo_dir.iterdir():
            if cell_dir.is_dir() and cell_dir.name.startswith('cell_'):
                numeric_id = cell_dir.name.split('_')[-1]
                numeric_ids_found.add(numeric_id)
                for f in cell_dir.glob(f"cell_{numeric_id}_*_features.npy"):
                    if match := re.search(
                        rf"cell_{numeric_id}_(\d+)_features\.npy", f.name
                    ):
                        timepoints.add(match.group(1))

    # Scan spharm for cell names and timepoints
    if spharm_dir.exists():
        cell_names = set()
        for cell_dir in spharm_dir.iterdir():
            if cell_dir.is_dir():
                cell_name = cell_dir.name
                # Only consider cells present in the name dictionary
                if cell_name in all_cell_names_from_dict:
                    cell_names.add(cell_name)
                    for f in cell_dir.glob(f"{cell_name}_*_l{SH_DEGREE}.npy"):
                        if match := re.search(
                            rf"{cell_name}_(\d+)_l{SH_DEGREE}\.npy", f.name
                        ):
                            timepoints.add(match.group(1))

    # Use all cell names from the dictionary as the definitive list, ensuring consistency
    # This handles cases where a cell might only have SH or only kinematic data
    # or might not have been found by scanning (e.g., empty directories)
    valid_cell_names = sorted(list(all_cell_names_from_dict))
    sorted_timepoints = sorted(list(timepoints), key=int) # Sort numerically

    print(f"Found {len(valid_cell_names)} unique cells (from name dict).")
    print(f"Found {len(sorted_timepoints)} unique timepoints.")

    if not valid_cell_names:
        print("Warning: No valid cells found. Check name dictionary and feature directories.")
    if not sorted_timepoints:
        print("Warning: No timepoints found. Check feature directories.")

    return valid_cell_names, sorted_timepoints

# --- Main Aggregation Logic ---
def main():
    print("Starting feature aggregation...")

    # 1. Load Name Dictionary
    name_to_id_str, id_str_to_name, all_cell_names_from_dict = load_name_dictionary(NAME_DICT_PATH)
    if name_to_id_str is None:
        return

    # 2. Discover Cells and Timepoints
    cell_names, timepoints = discover_cells_and_timepoints(GEO_FEATURES_DIR, SPHARM_DIR, all_cell_names_from_dict)
    if not cell_names or not timepoints:
        print("Cannot proceed without cells or timepoints.")
        return

    n_cells = len(cell_names)
    n_timepoints = len(timepoints)

    # 3. Create Mappings
    cell_to_row = {name: i for i, name in enumerate(cell_names)}
    time_to_col = {tp: j for j, tp in enumerate(timepoints)}

    # 4. Initialize Tensor
    feature_tensor = np.full((n_cells, n_timepoints, N_FEATURES), np.nan, dtype=np.float32)
    print(f"Initialized tensor with shape: {feature_tensor.shape}")

    # 5. Populate with Kinematic Features
    print("Populating kinematic features...")
    for cell_name in tqdm(cell_names, desc="Kinematic Features"):
        if cell_name not in name_to_id_str:
            # This cell might only exist in spharm, skip kinematic
            continue
            
        numeric_id = name_to_id_str[cell_name]
        row_idx = cell_to_row[cell_name]
        cell_geo_dir = GEO_FEATURES_DIR / f"cell_{numeric_id}"

        if not cell_geo_dir.exists():
            continue

        for timepoint in timepoints:
            col_idx = time_to_col[timepoint]
            feature_file = cell_geo_dir / f"cell_{numeric_id}_{timepoint}_features.npy"

            if feature_file.exists():
                try:
                    # Assuming the .npy file contains a dictionary
                    # Use allow_pickle=True as it might be a saved dictionary
                    data = np.load(feature_file, allow_pickle=True).item()
                    feature_vector = [data.get(fname, np.nan) for fname in KINEMATIC_FEATURES]
                    feature_tensor[row_idx, col_idx, :N_KINEMATIC] = feature_vector
                except Exception as e:
                    print(f"Warning: Could not load or process {feature_file}: {e}")

    # 6. Populate with SH Coefficients
    print("Populating Spherical Harmonic features...")
    for cell_name in tqdm(cell_names, desc="SH Features"):
        row_idx = cell_to_row[cell_name]
        cell_spharm_dir = SPHARM_DIR / cell_name

        if not cell_spharm_dir.exists():
            continue

        for timepoint in timepoints:
            col_idx = time_to_col[timepoint]
            sh_file = cell_spharm_dir / f"{cell_name}_{timepoint}_l{SH_DEGREE}.npy"

            if sh_file.exists():
                try:
                    # Assuming the .npy file contains a flat array of SH coefficients
                    sh_coeffs = np.load(sh_file)
                    if sh_coeffs.shape == (N_SH,):
                        feature_tensor[row_idx, col_idx, N_KINEMATIC:] = sh_coeffs
                    else:
                        print(f"Warning: SH file {sh_file} has unexpected shape {sh_coeffs.shape}. Expected ({N_SH},). Skipping.")
                except Exception as e:
                    print(f"Warning: Could not load or process {sh_file}: {e}")

    # 7. Save Results
    print(f"Saving aggregated tensor to {OUTPUT_TENSOR_PATH}")
    np.save(OUTPUT_TENSOR_PATH, feature_tensor)

    metadata = {
        'shape': feature_tensor.shape,
        'dimensions': ['cell', 'time', 'feature'],
        'cells': cell_names,
        'timepoints': timepoints,
        'features': ALL_FEATURES,
        'kinematic_features_indices': list(range(N_KINEMATIC)),
        'sh_features_indices': list(range(N_KINEMATIC, N_FEATURES)),
        'sh_degree': SH_DEGREE
    }

    print(f"Saving metadata to {OUTPUT_METADATA_PATH}")
    with open(OUTPUT_METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=4)

    print("Aggregation complete.")

if __name__ == "__main__":
    main() 
