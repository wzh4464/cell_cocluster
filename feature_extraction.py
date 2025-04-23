import numpy as np
import pandas as pd
from scipy import ndimage
from sklearn.decomposition import NMF
from read_data import load_nifti_file, get_all_nifti_files
from pathlib import Path

def calculate_cell_features(volume, label):
    """Calculate features for a single cell."""
    # Get cell mask
    cell_mask = (volume == label)
    
    # Calculate basic features
    volume = np.sum(cell_mask)
    surface_area = np.sum(ndimage.binary_dilation(cell_mask) & ~cell_mask)
    
    # Calculate centroid
    y, x, z = np.where(cell_mask)
    centroid = np.array([np.mean(x), np.mean(y), np.mean(z)])
    
    return {
        'volume': volume,
        'surface_area': surface_area,
        'centroid_x': centroid[0],
        'centroid_y': centroid[1],
        'centroid_z': centroid[2]
    }

def calculate_velocity(prev_centroid, curr_centroid):
    """Calculate velocity between two time points."""
    if prev_centroid is None:
        return np.zeros(3)
    return curr_centroid - prev_centroid

def calculate_acceleration(prev_velocity, curr_velocity):
    """Calculate acceleration between two time points."""
    if prev_velocity is None:
        return np.zeros(3)
    return curr_velocity - prev_velocity

def construct_feature_tensor(data_dir):
    """Construct feature tensor from NIfTI files."""
    nifti_files = get_all_nifti_files(data_dir)
    if not nifti_files:
        raise FileNotFoundError("No NIfTI files found in the data directory")
    
    # Initialize data structures
    all_features = []
    cell_trajectories = {}  # Store previous states for velocity/acceleration calculation
    
    # Process each time point
    for t, file_path in enumerate(nifti_files):
        volume = load_nifti_file(str(file_path))
        unique_labels = np.unique(volume)[1:]
        
        for label in unique_labels:
            # Calculate current features
            features = calculate_cell_features(volume, label)
            curr_centroid = np.array([features['centroid_x'], features['centroid_y'], features['centroid_z']])
            
            # Initialize or update cell trajectory
            if label not in cell_trajectories:
                cell_trajectories[label] = {
                    'prev_centroid': None,
                    'prev_velocity': None
                }
            
            # Calculate velocity and acceleration
            velocity = calculate_velocity(cell_trajectories[label]['prev_centroid'], curr_centroid)
            acceleration = calculate_acceleration(cell_trajectories[label]['prev_velocity'], velocity)
            
            # Update trajectory
            cell_trajectories[label]['prev_centroid'] = curr_centroid
            cell_trajectories[label]['prev_velocity'] = velocity
            
            # Add velocity and acceleration to features
            features.update({
                'velocity_x': velocity[0],
                'velocity_y': velocity[1],
                'velocity_z': velocity[2],
                'acceleration_x': acceleration[0],
                'acceleration_y': acceleration[1],
                'acceleration_z': acceleration[2],
                'time': t,
                'cell_id': label
            })
            
            all_features.append(features)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_features)
    
    # Calculate volume change rate
    df['volume_change_rate'] = df.groupby('cell_id')['volume'].diff()
    
    # Create feature matrix for NMF
    feature_columns = [
        'volume', 'surface_area', 'centroid_x', 'centroid_y', 'centroid_z',
        'velocity_x', 'velocity_y', 'velocity_z',
        'acceleration_x', 'acceleration_y', 'acceleration_z',
        'volume_change_rate'
    ]
    
    # Pivot the data to create the feature matrix
    feature_matrix = df.pivot(index='cell_id', columns='time')[feature_columns]
    
    return feature_matrix

def nmf_clustering(feature_matrix, n_components=5):
    """Perform NMF clustering on the feature matrix."""
    filled_matrix = feature_matrix.fillna(feature_matrix.mean())
    if filled_matrix.min().min() < 0:
        filled_matrix -= filled_matrix.min().min()
    
    model = NMF(n_components=n_components, init='random', random_state=0)
    W = model.fit_transform(filled_matrix)
    H = model.components_
    
    cluster_assignments = np.argmax(W, axis=1)
    
    return {
        'W': W,
        'H': H,
        'cluster_assignments': cluster_assignments
    }

def main():
    data_dir = "data"
    
    # Construct feature tensor
    print("Constructing feature tensor...")
    feature_matrix = construct_feature_tensor(data_dir)
    
    # Perform NMF clustering
    print("Performing NMF clustering...")
    results = nmf_clustering(feature_matrix)
    
    # Print results
    print("\nClustering Results:")
    print(f"Number of cells: {len(results['cluster_assignments'])}")
    print(f"Number of clusters: {len(np.unique(results['cluster_assignments']))}")
    print("\nCluster assignments:")
    for cell_id, cluster in enumerate(results['cluster_assignments']):
        print(f"Cell {cell_id}: Cluster {cluster}")

if __name__ == "__main__":
    main() 