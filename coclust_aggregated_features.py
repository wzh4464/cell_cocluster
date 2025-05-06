###
# File: ./coclust_aggregated_features.py
# Created Date: Tuesday, April 29th 2025
# Author: Zihan
# -----
# Last Modified: Friday, 2nd May 2025 10:14:11 am
# Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
# -----
# HISTORY:
# Date      		By   	Comments
# ----------		------	---------------------------------------------------------
###

# %%
import numpy as np
from pathlib import Path
import json
from sklearn.preprocessing import scale, StandardScaler
import cgc
from cgc.triclustering import Triclustering
import xarray as xr
import pandas as pd
from sklearn.cluster import SpectralCoclustering
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import warnings
from typing import Optional, Dict, List, Tuple, Any
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('clustering.log')
    ]
)
logger = logging.getLogger(__name__)

# Suppress specific warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in true_divide')

# --- Configuration ---
GEO_FEATURES_DIR = Path("DATA/geo_features")
SPHARM_DIR = Path("DATA/spharm")
NAME_DICT_PATH = Path("DATA/name_dictionary.csv")
TIMELINE_DATA_PATH = Path("cell_timeline_data.txt")
OUTPUT_TENSOR_PATH = Path("DATA/aggregated_features.npy")
OUTPUT_METADATA_PATH = Path("DATA/aggregated_metadata.json")
OUTPUT_SUBCLUSTER_DIR = Path("DATA/sub_triclusters")
SH_DEGREE = 15

class DataValidationError(Exception):
    """Custom exception for data validation errors."""
    pass

def validate_paths() -> None:
    """Validate existence of required paths and create output directory."""
    required_paths = {
        'Timeline Data': TIMELINE_DATA_PATH,
        'Output Tensor': OUTPUT_TENSOR_PATH,
        'Output Metadata': OUTPUT_METADATA_PATH
    }
    
    for name, path in required_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{name} not found at: {path}")
    
    OUTPUT_SUBCLUSTER_DIR.mkdir(parents=True, exist_ok=True)

def load_timeline_data(data_file: Path = TIMELINE_DATA_PATH) -> Optional[pd.DataFrame]:
    """Load and validate timeline data."""
    try:
        df = pd.read_csv(data_file, sep='\t', index_col=0)
        df.columns = df.columns.astype(str)
        
        # Validate data
        if df.empty:
            raise DataValidationError("Timeline data is empty")
        
        if df.isnull().values.any():
            logger.warning(f"Timeline data contains {df.isnull().sum().sum()} NaN values")
        
        logger.info(f"Loaded timeline data with shape: {df.shape}")
        return df
    
    except Exception as e:
        logger.error(f"Error loading timeline data: {str(e)}")
        return None

def preprocess_tensor(tensor: np.ndarray, features: Optional[List[str]] = None) -> np.ndarray:
    """Enhanced tensor preprocessing with better error handling and validation."""
    try:
        # Input validation
        if not isinstance(tensor, np.ndarray):
            raise TypeError("Input must be a numpy array")

        if tensor.ndim != 3:
            raise ValueError(f"Expected 3D tensor, got {tensor.ndim}D")

        # Memory efficient copy
        tensor = tensor.astype(np.float64, copy=True)

        # Check for invalid values
        n_nan = np.isnan(tensor).sum()
        n_zeros = (tensor == 0).sum()
        
        if n_nan > 0:
            logger.warning(f"Found {n_nan} NaN values in tensor")
        if n_zeros > 0:
            logger.warning(f"Found {n_zeros} zero values in tensor")

        # Reshape for feature-wise processing
        original_shape = tensor.shape
        tensor_2d = tensor.reshape(-1, original_shape[2])

        # Create mask for valid values (non-zero and non-nan)
        valid_mask = ~np.isnan(tensor_2d) & (tensor_2d != 0)

        # Process each feature separately
        for feature_idx in range(tensor_2d.shape[1]):
            feature_data = tensor_2d[:, feature_idx]
            valid_data = feature_data[valid_mask[:, feature_idx]]
            
            if len(valid_data) > 0:  # If we have valid data points
                # Calculate mean and std from valid data only
                feature_mean = np.mean(valid_data)
                feature_std = np.std(valid_data)
                
                if feature_std > 1e-10:  # Check for near-zero variance
                    # Standardize the entire feature column
                    standardized_data = np.where(
                        valid_mask[:, feature_idx],
                        (feature_data - feature_mean) / feature_std,
                        0
                    )
                    
                    # Shift data to be positive (required for log-based clustering)
                    min_val = np.min(standardized_data)
                    if min_val < 0:
                        shift = -min_val + 1  # Add 1 to ensure all values are > 0
                        standardized_data = standardized_data + shift
                    
                    tensor_2d[:, feature_idx] = standardized_data
                else:
                    logger.warning(f"Feature {feature_idx} has near-zero variance after removing zeros and NaNs")
                    tensor_2d[:, feature_idx] = 1  # Set to 1 instead of 0 for log-based clustering
            else:
                logger.warning(f"Feature {feature_idx} has no valid data points after removing zeros and NaNs")
                tensor_2d[:, feature_idx] = 1  # Set to 1 instead of 0 for log-based clustering

        # Reshape back to original dimensions
        tensor_scaled = tensor_2d.reshape(original_shape)

        # Add small epsilon to zeros to avoid log(0)
        epsilon = 1e-10
        tensor_scaled = np.where(tensor_scaled == 0, epsilon, tensor_scaled)

        # Validate output
        if not np.isfinite(tensor_scaled).all():
            raise ValueError("Scaling produced non-finite values")
        
        if np.any(tensor_scaled <= 0):
            raise ValueError("Scaling produced non-positive values")

        return tensor_scaled

    except Exception as e:
        logger.error(f"Error in tensor preprocessing: {str(e)}")
        raise

def perform_biclustering(data: pd.DataFrame, n_clusters: int = 9) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Enhanced bi-clustering with better validation and error handling."""
    if data is None:
        logger.error("No data provided for bi-clustering")
        return None, None

    try:
        logger.info(f"Starting bi-clustering with {n_clusters} clusters...") # sourcery skip: extract-method
        matrix = data.values

        # Validate input
        if np.all(np.isnan(matrix)):
            raise ValueError("Input matrix contains only NaN values")

        if np.all(matrix == 0):
            raise ValueError("Input matrix contains only zero values")

        # Handle NaN/Inf values
        if not np.isfinite(matrix).all():
            logger.warning("Replacing non-finite values with 0")
            matrix = np.nan_to_num(matrix)

        # Perform clustering
        model = SpectralCoclustering(n_clusters=n_clusters, random_state=0)
        model.fit(matrix)

        # Create DataFrames for clustering results
        cell_cluster_df = pd.DataFrame({
            'cell': data.index,
            'cluster': model.row_labels_
        })

        time_cluster_df = pd.DataFrame({
            'time': data.columns,
            'cluster': model.column_labels_
        })

        # Sort by cluster and then by name/time
        cell_cluster_df = cell_cluster_df.sort_values(['cluster', 'cell'])
        time_cluster_df = time_cluster_df.sort_values(['cluster', 'time'])

        # Reorder the matrix based on the sorted indices
        reordered_matrix = matrix[cell_cluster_df.index]
        reordered_matrix = reordered_matrix[:, time_cluster_df.index]

        from cocluster_timeline import plot_coclustered_matrix

        plot_coclustered_matrix(
            matrix=reordered_matrix,
            cluster_df=cell_cluster_df,
            time_df=time_cluster_df,
            output_file=OUTPUT_SUBCLUSTER_DIR / "bi_clustering.png"
        )
        
        logger.info(f"Bi-clustering complete: {len(cell_cluster_df['cluster'].unique())} cell clusters, {len(time_cluster_df['cluster'].unique())} time clusters")

        return cell_cluster_df, time_cluster_df

    except Exception as e:
        logger.error(f"Error in bi-clustering: {str(e)}")
        return None, None

def process_subtensor(args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Process a single sub-tensor for parallel execution."""
    sub_tensor = args['sub_tensor']
    sub_metadata = args['sub_metadata']
    n_cell_clusters = args['n_cell_clusters']
    n_time_clusters = args['n_time_clusters']
    n_feature_clusters = args['n_feature_clusters']
    bicluster_ids = args['bicluster_ids']
    
    try:
        cell_bic_id, time_bic_id = bicluster_ids
        logger.info(f"Processing sub-tensor for bi-clusters (Cell: {cell_bic_id}, Time: {time_bic_id})")
        logger.info(f"Sub-tensor shape before preprocessing: {sub_tensor.shape}")
        
        # Validate dimensions
        min_dims = (n_cell_clusters, n_time_clusters, n_feature_clusters)
        if any(s < m for s, m in zip(sub_tensor.shape, min_dims)):
            logger.warning(f"Sub-tensor too small: shape={sub_tensor.shape}, required={min_dims}")
            return None
        
        # Process tensor
        processed_sub_tensor = preprocess_tensor(sub_tensor)
        logger.info(f"Sub-tensor shape after preprocessing: {processed_sub_tensor.shape}")
        
        # Check if tensor contains only zeros after preprocessing
        if np.all(processed_sub_tensor == 0):
            logger.error("Processed tensor contains only zeros!")
            return None
            
        # Check if tensor contains any NaN values
        if np.any(np.isnan(processed_sub_tensor)):
            logger.error("Processed tensor contains NaN values!")
            return None
            
        # Print some statistics about the processed tensor
        logger.info(f"Processed tensor stats - Mean: {np.mean(processed_sub_tensor):.4f}, "
                   f"Std: {np.std(processed_sub_tensor):.4f}, "
                   f"Min: {np.min(processed_sub_tensor):.4f}, "
                   f"Max: {np.max(processed_sub_tensor):.4f}")
        
        # Transpose tensor for cgc input (time, cell, feature)
        tensor_for_cgc = processed_sub_tensor.transpose(1, 0, 2)
        logger.info(f"Tensor shape after transpose: {tensor_for_cgc.shape}")
        
        # Perform tri-clustering with more iterations and stricter convergence
        tc = Triclustering(
            tensor_for_cgc,
            n_time_clusters,
            n_cell_clusters,
            n_feature_clusters,
            max_iterations=1000,  # Increased from 500
            conv_threshold=0.01,  # More strict convergence
            nruns=5  # Increased from 3
        )
        
        results = tc.run_with_threads(nthreads=1)
        
        # Validate clustering results
        if results is None:
            logger.error("Triclustering failed to produce results")
            return None
            
        # Check if we got meaningful clusters
        if len(np.unique(results.row_clusters)) == 1 or \
           len(np.unique(results.col_clusters)) == 1 or \
           len(np.unique(results.bnd_clusters)) == 1:
            logger.error("Triclustering produced degenerate clusters")
            logger.info(f"Unique clusters - Time: {np.unique(results.row_clusters)}, "
                       f"Cell: {np.unique(results.col_clusters)}, "
                       f"Feature: {np.unique(results.bnd_clusters)}")
            return None
            
        logger.info(f"Clustering complete - Time clusters: {np.unique(results.row_clusters)}, "
                   f"Cell clusters: {np.unique(results.col_clusters)}, "
                   f"Feature clusters: {np.unique(results.bnd_clusters)}")
        
        return {
            'time_labels': results.row_clusters.tolist(),
            'cell_labels': results.col_clusters.tolist(),
            'feature_labels': results.bnd_clusters.tolist(),
            'metadata': sub_metadata,
            'bicluster_ids': bicluster_ids,
            'tensor_stats': {
                'mean': float(np.mean(processed_sub_tensor)),
                'std': float(np.std(processed_sub_tensor)),
                'min': float(np.min(processed_sub_tensor)),
                'max': float(np.max(processed_sub_tensor)),
                'zeros_percentage': float(np.mean(processed_sub_tensor == 0) * 100)
            }
        }
        
    except Exception as e:
        logger.error(f"Error processing sub-tensor {bicluster_ids}: {str(e)}")
        return None

def visualize_coclusters(main_tensor: np.ndarray, 
                        cell_bicluster_df: pd.DataFrame, 
                        time_bicluster_df: pd.DataFrame,
                        main_metadata: Dict,
                        output_dir: Path) -> None:
    """
    Visualize the coclusters projection on the original tensor.
    
    Args:
        main_tensor: The original 3D tensor (cells x timepoints x features)
        cell_bicluster_df: DataFrame with cell cluster assignments
        time_bicluster_df: DataFrame with time cluster assignments
        main_metadata: Dictionary containing metadata about cells and timepoints
        output_dir: Directory to save the visualization plots
    """
    try:
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Create cell x time projection matrix
        projection = np.mean(main_tensor, axis=2)  # Average over features
        
        # 2. Create cluster assignment matrices
        n_cells = len(main_metadata['cells'])
        n_timepoints = len(main_metadata['timepoints'])
        
        # Create mapping dictionaries
        cell_to_cluster = dict(zip(cell_bicluster_df['cell'], cell_bicluster_df['cluster']))
        time_to_cluster = dict(zip(time_bicluster_df['time'], time_bicluster_df['cluster']))
        
        # Create cluster matrices
        cluster_matrix = np.zeros((n_cells, n_timepoints))
        value_matrix = projection.copy()
        
        # Fill matrices
        for i, cell in enumerate(main_metadata['cells']):
            for j, time in enumerate(main_metadata['timepoints']):
                if cell in cell_to_cluster and str(time) in time_to_cluster:
                    cell_cluster = cell_to_cluster[cell]
                    time_cluster = time_to_cluster[str(time)]
                    cluster_matrix[i, j] = cell_cluster * time_bicluster_df['cluster'].nunique() + time_cluster
        
        # 3. Create visualizations
        plt.figure(figsize=(20, 10))
        
        # Plot 1: Original projection
        plt.subplot(121)
        sns.heatmap(value_matrix, 
                   cmap='viridis', 
                   xticklabels=main_metadata['timepoints'][::5],  # Show every 5th label
                   yticklabels=main_metadata['cells'][::5],
                   cbar_kws={'label': 'Average Feature Value'})
        plt.title('Original Data Projection\n(Averaged over features)')
        plt.xlabel('Timepoints')
        plt.ylabel('Cells')
        
        # Plot 2: Cluster assignments
        plt.subplot(122)
        # Create a custom colormap for clusters
        n_clusters = int(cluster_matrix.max() + 1)
        colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))
        cmap = LinearSegmentedColormap.from_list('custom', colors)
        
        sns.heatmap(cluster_matrix,
                   cmap=cmap,
                   xticklabels=main_metadata['timepoints'][::5],
                   yticklabels=main_metadata['cells'][::5],
                   cbar_kws={'label': 'Cluster ID'})
        plt.title('Cocluster Assignments')
        plt.xlabel('Timepoints')
        plt.ylabel('Cells')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(output_dir / 'cocluster_projection.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Create individual cluster visualizations
        unique_clusters = np.unique(cluster_matrix)
        plt.figure(figsize=(15, 5 * ((len(unique_clusters) + 2) // 3)))
        
        for idx, cluster_id in enumerate(unique_clusters):
            plt.subplot(((len(unique_clusters) + 2) // 3), 3, idx + 1)
            mask = cluster_matrix != cluster_id
            masked_values = np.ma.array(value_matrix, mask=mask)
            
            sns.heatmap(masked_values,
                       cmap='viridis',
                       xticklabels=main_metadata['timepoints'][::5],
                       yticklabels=False,
                       cbar_kws={'label': 'Value'})
            plt.title(f'Cluster {int(cluster_id)}')
            plt.xlabel('Timepoints')
            plt.ylabel('Cells')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'individual_clusters.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualization saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error in visualization: {str(e)}")
        raise

def visualize_triclusters(results_path: Path, output_dir: Path) -> None:
    """
    Visualize the results of triclustering, projecting to cell-time plane.
    
    Args:
        results_path: Path to the JSON file containing all subcluster results
        output_dir: Directory to save visualization outputs
    """
    try:
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load results
        with open(results_path, 'r') as f:
            all_results = json.load(f)
        
        if not all_results:
            logger.error("No tricluster results found")
            return
            
        # Specify the tricluster IDs to visualize
        target_triclusters = [('0', '0'), ('6', '6'), ('7', '6'), ('7', '4')]
        logger.info(f"Only generating plots for specific triclusters: {target_triclusters}")
        
        # Create data structure to hold results for later plotting in a 2x2 grid
        plot_data = {}
        
        # Process each subcluster
        for key, result in all_results.items():
            if not result or 'cell_labels' not in result or 'time_labels' not in result:
                logger.warning(f"Skipping invalid result for {key}")
                continue
                
            # Extract metadata and labels
            metadata = result['metadata']
            cell_labels = result['cell_labels'] 
            time_labels = result['time_labels']
            bicluster_ids = result['bicluster_ids']
            
            # Skip if not one of our target triclusters
            if (str(bicluster_ids[0]), str(bicluster_ids[1])) not in target_triclusters:
                logger.info(f"Skipping tricluster {bicluster_ids} (not in target list)")
                continue
                
            logger.info(f"Processing target tricluster {bicluster_ids}")
            
            # Check for required data
            if not metadata or 'cells' not in metadata or 'timepoints' not in metadata:
                logger.warning(f"Missing metadata for {key}")
                continue
                
            # Create mapping from indices to names
            cells = metadata['cells']
            times = metadata['timepoints']
            
            # Check if cell_labels and cells have the same length
            if len(cell_labels) != len(cells):
                logger.warning(f"Cell labels length ({len(cell_labels)}) doesn't match cells length ({len(cells)}) for {key}")
                # We need to ensure these have the same length
                min_len = min(len(cell_labels), len(cells))
                cells = cells[:min_len]
                cell_labels = cell_labels[:min_len]
            
            # Check if time_labels and times have the same length
            if len(time_labels) != len(times):
                logger.warning(f"Time labels length ({len(time_labels)}) doesn't match times length ({len(times)}) for {key}")
                # We need to ensure these have the same length
                min_len = min(len(time_labels), len(times))
                times = times[:min_len]
                time_labels = time_labels[:min_len]
            
            # Create DataFrames for the clustering results
            cell_cluster_df = pd.DataFrame({
                'cell': cells,
                'cluster': cell_labels
            })
            
            time_cluster_df = pd.DataFrame({
                'time': times,
                'cluster': time_labels
            })
            
            # Sort by cluster and then by name/time
            cell_cluster_df = cell_cluster_df.sort_values(['cluster', 'cell'])
            time_cluster_df = time_cluster_df.sort_values(['cluster', 'time'])
            
            # Create cell-time projection matrix
            # We'll use a binary matrix to show cluster assignments
            matrix = np.zeros((len(cells), len(times)))
            
            # Fill the matrix based on cluster assignments
            for i, cell_idx in enumerate(range(len(cells))):
                for j, time_idx in enumerate(range(len(times))):
                    # Assign a value based on cell and time cluster
                    cell_cluster = cell_labels[cell_idx]
                    time_cluster = time_labels[time_idx]
                    # Create a unique cluster index
                    cluster_id = cell_cluster * 100 + time_cluster
                    matrix[i, j] = cluster_id
            
            # Create lists for reordering
            cell_indices = []
            for cell in cell_cluster_df['cell']:
                try:
                    cell_indices.append(cells.index(cell))
                except ValueError:
                    # Skip if cell not found
                    logger.warning(f"Cell {cell} not found in cells list")
                    continue
            
            time_indices = []
            for time in time_cluster_df['time']:
                try:
                    time_indices.append(times.index(str(time)))
                except ValueError:
                    try:
                        # Try without string conversion
                        time_indices.append(times.index(time))
                    except ValueError:
                        # Skip if time not found
                        logger.warning(f"Time {time} not found in times list")
                        continue
            
            if not cell_indices or not time_indices:
                logger.warning(f"No valid indices for {key}")
                continue
                
            # Safety check for indices
            cell_indices = [idx for idx in cell_indices if 0 <= idx < len(cells)]
            time_indices = [idx for idx in time_indices if 0 <= idx < len(times)]
            
            if not cell_indices or not time_indices:
                logger.warning(f"No valid indices after bounds check for {key}")
                continue
                
            reordered_matrix = matrix[np.ix_(cell_indices, time_indices)]
            
            # Calculate unique cluster combinations
            unique_clusters = np.unique(reordered_matrix)
            n_clusters = len(unique_clusters)
            
            # Save plot data for 2x2 grid
            plot_data[f"C{bicluster_ids[0]}_T{bicluster_ids[1]}"] = {
                'matrix': reordered_matrix,
                'n_clusters': n_clusters,
                'cell_sizes': cell_cluster_df['cluster'].value_counts().sort_index(),
                'time_sizes': time_cluster_df['cluster'].value_counts().sort_index(),
                'cell_indices': cell_indices,
                'time_indices': time_indices,
                'cell_df': cell_cluster_df,
                'time_df': time_cluster_df
            }
            
            # Also save individual plots for reference
            individual_plot_path = output_dir / f"tricluster_C{bicluster_ids[0]}_T{bicluster_ids[1]}.png"
            save_individual_plot(
                reordered_matrix, 
                n_clusters,
                cell_cluster_df['cluster'].value_counts().sort_index(),
                time_cluster_df['cluster'].value_counts().sort_index(),
                cell_indices,
                time_indices,
                cell_cluster_df,
                time_cluster_df,
                bicluster_ids, 
                individual_plot_path
            )
            
        # Create a 2x2 grid of the selected triclusters
        if len(plot_data) == 4:
            create_grid_plot(plot_data, output_dir)
        else:
            logger.warning(f"Expected 4 triclusters for 2x2 grid, but found {len(plot_data)}")
        
        logger.info(f"Saved selected tricluster visualizations to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error in tricluster visualization: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise
        
def save_individual_plot(reordered_matrix, n_clusters, cell_sizes, time_sizes, 
                         cell_indices, time_indices, cell_cluster_df, time_cluster_df,
                         bicluster_ids, output_path):
    """Helper function to save an individual tricluster plot"""
    plt.figure(figsize=(12, 8))
    
    # Generate a good colormap
    cmap = plt.cm.get_cmap('tab20', max(n_clusters, 1))
    
    # Plot the reordered matrix
    im = plt.imshow(reordered_matrix, cmap=cmap, aspect='auto')
    plt.colorbar(im, label='Cluster ID')
    
    # Add cluster boundaries
    # Add vertical lines for column clusters
    col_cumsum = np.cumsum(time_sizes)
    for x in col_cumsum[:-1]:
        if 0 < x < len(time_indices):  # Check bounds
            plt.axvline(x=x, color='black', linestyle='--', linewidth=0.7)
    
    # Add horizontal lines for row clusters
    row_cumsum = np.cumsum(cell_sizes)
    for y in row_cumsum[:-1]:
        if 0 < y < len(cell_indices):  # Check bounds
            plt.axhline(y=y, color='black', linestyle='--', linewidth=0.7)
    
    # Set labels and title
    plt.title(f'Tricluster Projection - Cell-Time Plane\nBi-cluster: Cell {bicluster_ids[0]}, Time {bicluster_ids[1]}')
    plt.xlabel('Time Points')
    plt.ylabel('Cells')
    
    # Set x-axis ticks with safety checks
    x_tick_freq = max(1, len(time_indices) // 10)
    x_ticks = np.arange(0, len(time_indices), x_tick_freq)
    x_labels = []
    for i in x_ticks:
        if i < len(time_indices) and time_indices[i] < len(time_cluster_df):
            x_labels.append(time_cluster_df['time'].iloc[time_indices[i]])
        else:
            x_labels.append('')
    plt.xticks(x_ticks, x_labels, rotation=45)
    
    # Set y-axis ticks with safety checks
    y_tick_freq = max(1, len(cell_indices) // 15)
    y_ticks = np.arange(0, len(cell_indices), y_tick_freq)
    y_labels = []
    for i in y_ticks:
        if i < len(cell_indices) and cell_indices[i] < len(cell_cluster_df):
            y_labels.append(cell_cluster_df['cell'].iloc[cell_indices[i]])
        else:
            y_labels.append('')
    plt.yticks(y_ticks, y_labels)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_grid_plot(plot_data, output_dir):
    """Create a 2x2 grid of tricluster plots"""
    plt.figure(figsize=(24, 20))
    
    # Define the order for the grid
    grid_order = ['C0_T0', 'C6_T6', 'C7_T6', 'C7_T4']
    
    for idx, key in enumerate(grid_order):
        if key not in plot_data:
            logger.warning(f"Missing data for {key}, cannot create complete grid")
            return
            
        data = plot_data[key]
        
        plt.subplot(2, 2, idx+1)
        
        # Generate a good colormap
        cmap = plt.cm.get_cmap('tab20', max(data['n_clusters'], 1))
        
        # Plot the reordered matrix
        im = plt.imshow(data['matrix'], cmap=cmap, aspect='auto')
        
        # Add cluster boundaries
        # Add vertical lines for column clusters
        col_cumsum = np.cumsum(data['time_sizes'])
        for x in col_cumsum[:-1]:
            if 0 < x < len(data['time_indices']):  # Check bounds
                plt.axvline(x=x, color='black', linestyle='--', linewidth=0.7)
        
        # Add horizontal lines for row clusters
        row_cumsum = np.cumsum(data['cell_sizes'])
        for y in row_cumsum[:-1]:
            if 0 < y < len(data['cell_indices']):  # Check bounds
                plt.axhline(y=y, color='black', linestyle='--', linewidth=0.7)
        
        # Set title and labels
        plt.title(f'Tricluster {key}', fontsize=14)
        plt.xlabel('Time Points', fontsize=12)
        plt.ylabel('Cells', fontsize=12)
        
        # Add ticks with labels for better readability
        # Set x-axis ticks with safety checks
        x_tick_freq = max(1, len(data['time_indices']) // 8)  # Fewer ticks for grid view
        x_ticks = np.arange(0, len(data['time_indices']), x_tick_freq)
        x_labels = []
        for i in x_ticks:
            if i < len(data['time_indices']) and data['time_indices'][i] < len(data['time_df']):
                x_labels.append(data['time_df']['time'].iloc[data['time_indices'][i]])
            else:
                x_labels.append('')
        plt.xticks(x_ticks, x_labels, rotation=45, fontsize=10)
        
        # Set y-axis ticks with safety checks
        y_tick_freq = max(1, len(data['cell_indices']) // 10)  # Fewer ticks for grid view
        y_ticks = np.arange(0, len(data['cell_indices']), y_tick_freq)
        y_labels = []
        for i in y_ticks:
            if i < len(data['cell_indices']) and data['cell_indices'][i] < len(data['cell_df']):
                y_labels.append(data['cell_df']['cell'].iloc[data['cell_indices'][i]])
            else:
                y_labels.append('')
        plt.yticks(y_ticks, y_labels, fontsize=10)
    
    plt.suptitle('Tricluster Comparison (2x2 Grid)', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Make room for the suptitle
    grid_plot_path = output_dir / "tricluster_grid_2x2.png"
    plt.savefig(grid_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_triclusters_combined(results_path: Path, output_dir: Path, main_tensor: np.ndarray, 
                             cell_bicluster_df: pd.DataFrame, time_bicluster_df: pd.DataFrame,
                             main_metadata: Dict) -> None:
    """
    Visualize all tricluster results combined on the main tensor's cell-time projection.
    
    Args:
        results_path: Path to the JSON file containing all subcluster results
        output_dir: Directory to save visualization outputs
        main_tensor: The original 3D tensor (cells x timepoints x features)
        cell_bicluster_df: DataFrame with cell bicluster assignments
        time_bicluster_df: DataFrame with time bicluster assignments
        main_metadata: Dictionary containing metadata about cells and timepoints
    """
    try:
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load tricluster results
        with open(results_path, 'r') as f:
            all_results = json.load(f)
        
        if not all_results:
            logger.error("No tricluster results found")
            return
            
        # Create mappings for cell and time indices in main tensor
        cell_to_idx = {cell: idx for idx, cell in enumerate(main_metadata['cells'])}
        time_to_idx = {str(time): idx for idx, time in enumerate(main_metadata['timepoints'])}
        
        # Get bicluster assignments for all cells and timepoints
        cell_bicluster_map = dict(zip(cell_bicluster_df['cell'], cell_bicluster_df['cluster']))
        time_bicluster_map = dict(zip(time_bicluster_df['time'], time_bicluster_df['cluster']))
        
        # Sort cells and times by bicluster
        sorted_cells = cell_bicluster_df.sort_values(['cluster', 'cell'])['cell'].tolist()
        sorted_times = time_bicluster_df.sort_values(['cluster', 'time'])['time'].tolist()
        
        # Create cell and time indices for the sorted order
        cell_indices = [cell_to_idx[cell] for cell in sorted_cells if cell in cell_to_idx]
        time_indices = [time_to_idx[str(time)] for time in sorted_times if str(time) in time_to_idx]
        
        # Create a 2D matrix for the main projection (cell-time)
        projection = np.mean(main_tensor, axis=2)  # Average over features
        
        # Create a matrix to store tricluster assignments
        tricluster_matrix = np.full((len(main_metadata['cells']), len(main_metadata['timepoints'])), -1, dtype=float)
        
        # Track which cells and times have been assigned a tricluster
        assigned_cells = set()
        assigned_times = set()
        
        # Create a colormap for different triclusters
        # Count total unique tricluster combinations
        unique_triclusters = set()
        
        # Process each tricluster result
        for key, result in all_results.items():
            if not result or 'bicluster_ids' not in result:
                continue
                
            cell_bic_id, time_bic_id = result['bicluster_ids']
            if 'cell_labels' not in result or 'time_labels' not in result or 'metadata' not in result:
                continue
                
            # Extract data
            metadata = result['metadata']
            cell_labels = result['cell_labels'] 
            time_labels = result['time_labels']
            
            if not metadata or 'cells' not in metadata or 'timepoints' not in metadata:
                continue
                
            # Get cells and times for this tricluster
            sub_cells = metadata['cells']
            sub_times = metadata['timepoints']
            
            # Check array lengths
            min_cell_len = min(len(sub_cells), len(cell_labels))
            min_time_len = min(len(sub_times), len(time_labels))
            
            # Record unique tricluster combinations
            for c_label in range(max(cell_labels) + 1):
                for t_label in range(max(time_labels) + 1):
                    unique_triclusters.add((cell_bic_id, time_bic_id, c_label, t_label))
        
        # Create color mapping
        n_unique_triclusters = len(unique_triclusters)
        logger.info(f"Found {n_unique_triclusters} unique tricluster combinations")
        
        # Use a good colormap with enough colors
        if n_unique_triclusters <= 20:
            cmap_name = 'tab20'
        else:
            cmap_name = 'viridis'
        
        cmap = plt.cm.get_cmap(cmap_name, max(n_unique_triclusters, 1))
        tricluster_to_color = {}
        
        for i, (cell_bic, time_bic, cell_sub, time_sub) in enumerate(sorted(unique_triclusters)):
            tricluster_to_color[(cell_bic, time_bic, cell_sub, time_sub)] = i
        
        # Create assigned matrix to track assignment
        is_assigned = np.zeros((len(main_metadata['cells']), len(main_metadata['timepoints'])), dtype=bool)
        
        # Fill tricluster matrix with assignments
        for key, result in all_results.items():
            if not result or 'bicluster_ids' not in result:
                continue
                
            cell_bic_id, time_bic_id = result['bicluster_ids']
            if 'cell_labels' not in result or 'time_labels' not in result or 'metadata' not in result:
                continue
                
            # Extract data
            metadata = result['metadata']
            cell_labels = result['cell_labels'] 
            time_labels = result['time_labels']
            
            if not metadata or 'cells' not in metadata or 'timepoints' not in metadata:
                continue
                
            # Get cells and times for this tricluster
            sub_cells = metadata['cells']
            sub_times = metadata['timepoints']
            
            # Check array lengths
            min_cell_len = min(len(sub_cells), len(cell_labels))
            min_time_len = min(len(sub_times), len(time_labels))
            
            # Assign tricluster IDs to the matrix
            for c_idx in range(min_cell_len):
                cell = sub_cells[c_idx]
                if cell not in cell_to_idx:
                    continue
                    
                main_c_idx = cell_to_idx[cell]
                c_label = cell_labels[c_idx]
                
                for t_idx in range(min_time_len):
                    time = sub_times[t_idx]
                    time_tri_id = time_labels[t_idx]
                    
                    if str(time) not in time_to_idx:
                        continue
                        
                    main_t_idx = time_to_idx[str(time)]
                    
                    # Skip if already assigned
                    if is_assigned[main_c_idx, main_t_idx]:
                        continue
                        
                    # Assign tricluster ID as color index
                    tricluster_id = tricluster_to_color.get((cell_bic_id, time_bic_id, c_label, time_tri_id), -1)
                    if tricluster_id != -1:
                        tricluster_matrix[main_c_idx, main_t_idx] = tricluster_id
                        is_assigned[main_c_idx, main_t_idx] = True
                        
                        # Record that this cell and time have been assigned
                        assigned_cells.add(cell)
                        assigned_times.add(time)
        
        # Reorder the matrix based on biclustering
        reordered_projection = projection[np.ix_(cell_indices, time_indices)]
        reordered_tricluster = tricluster_matrix[np.ix_(cell_indices, time_indices)]
        
        # Create masks for visualization
        masked_tricluster = np.ma.masked_where(reordered_tricluster < 0, reordered_tricluster)
        
        # Create a figure for visualization
        plt.figure(figsize=(20, 12))
        
        # Plot original data projection
        plt.subplot(121)
        plt.imshow(reordered_projection, cmap='viridis', aspect='auto')
        plt.colorbar(label='Average Feature Value')
        plt.title('Main Tensor Projection\n(Cell-Time plane, reordered by biclusters)')
        plt.xlabel('Time Points')
        plt.ylabel('Cells')
        
        # Add bicluster boundaries
        cell_sizes = cell_bicluster_df['cluster'].value_counts().sort_index()
        time_sizes = time_bicluster_df['cluster'].value_counts().sort_index()
        
        # Add vertical lines for column clusters
        col_cumsum = np.cumsum(time_sizes)
        for x in col_cumsum[:-1]:
            if 0 < x < len(time_indices):
                plt.axvline(x=x, color='white', linestyle='--', linewidth=0.7)
        
        # Add horizontal lines for row clusters
        row_cumsum = np.cumsum(cell_sizes)
        for y in row_cumsum[:-1]:
            if 0 < y < len(cell_indices):
                plt.axhline(y=y, color='white', linestyle='--', linewidth=0.7)
        
        # Set ticks
        x_tick_freq = max(1, len(time_indices) // 10)
        x_ticks = np.arange(0, len(time_indices), x_tick_freq)
        x_labels = [sorted_times[i] if i < len(sorted_times) else '' for i in x_ticks]
        plt.xticks(x_ticks, x_labels, rotation=45)
        
        y_tick_freq = max(1, len(cell_indices) // 15)
        y_ticks = np.arange(0, len(cell_indices), y_tick_freq)
        y_labels = [sorted_cells[i] if i < len(sorted_cells) else '' for i in y_ticks]
        plt.yticks(y_ticks, y_labels)
        
        # Plot tricluster assignments
        plt.subplot(122)
        im = plt.imshow(masked_tricluster, cmap=cmap, aspect='auto')
        plt.title('Tricluster Assignments\n(Cell-Time plane, reordered by biclusters)')
        plt.xlabel('Time Points')
        plt.ylabel('Cells')
        
        # Add bicluster boundaries
        for x in col_cumsum[:-1]:
            if 0 < x < len(time_indices):
                plt.axvline(x=x, color='white', linestyle='--', linewidth=0.7)
        
        for y in row_cumsum[:-1]:
            if 0 < y < len(cell_indices):
                plt.axhline(y=y, color='white', linestyle='--', linewidth=0.7)
        
        # Set ticks
        plt.xticks(x_ticks, x_labels, rotation=45)
        plt.yticks(y_ticks, y_labels)
        
        # Save the visualization
        plt.tight_layout()
        combined_plot_path = output_dir / "combined_tricluster_view.png"
        plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create a legend plot to show the tricluster mapping
        if n_unique_triclusters > 0:
            legend_height = max(8, min(30, n_unique_triclusters // 4))
            plt.figure(figsize=(10, legend_height))
            
            # Create patches for legend
            from matplotlib.patches import Patch
            legend_elements = []
            
            for (cell_bic, time_bic, cell_sub, time_sub), color_idx in tricluster_to_color.items():
                color = cmap(color_idx)
                label = f"C{cell_bic}-T{time_bic}: Sub(C{cell_sub}-T{time_sub})"
                legend_elements.append(Patch(facecolor=color, label=label))
            
            # Sort by label for better readability
            legend_elements.sort(key=lambda x: x.get_label())
            
            # Create legend plot
            plt.axis('off')  # Turn off axis
            plt.legend(handles=legend_elements, loc='center', ncol=3, 
                      fontsize='small', title="Tricluster Legend")
            
            legend_plot_path = output_dir / "tricluster_legend.png"
            plt.savefig(legend_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        # Report statistics
        logger.info(f"Combined visualization saved to {combined_plot_path}")
        logger.info(f"Assigned {len(assigned_cells)} cells and {len(assigned_times)} timepoints to triclusters")
        logger.info(f"Coverage: {np.sum(is_assigned) / is_assigned.size * 100:.2f}% of the cell-time matrix")
        
    except Exception as e:
        logger.error(f"Error in combined tricluster visualization: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

def visualize_triclusters_embedded(results_path: Path, output_dir: Path, main_tensor: np.ndarray, 
                           cell_bicluster_df: pd.DataFrame, time_bicluster_df: pd.DataFrame,
                           main_metadata: Dict) -> None:
    """
    Visualize tricluster results by embedding them directly in the main tensor projection,
    similar to how cocluster_timeline.py visualizes results.
    
    Args:
        results_path: Path to the JSON file containing all subcluster results
        output_dir: Directory to save visualization outputs
        main_tensor: The original 3D tensor (cells x timepoints x features)
        cell_bicluster_df: DataFrame with cell bicluster assignments
        time_bicluster_df: DataFrame with time bicluster assignments
        main_metadata: Dictionary containing metadata about cells and timepoints
    """
    try:
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load tricluster results
        with open(results_path, 'r') as f:
            all_results = json.load(f)
        
        if not all_results:
            logger.error("No tricluster results found")
            return
            
        # Get number of biclusters
        n_cell_biclusters = cell_bicluster_df['cluster'].nunique()
        n_time_biclusters = time_bicluster_df['cluster'].nunique()
        
        logger.info(f"Found {n_cell_biclusters} cell biclusters and {n_time_biclusters} time biclusters")
        
        # Create mappings for cell and time indices and biclusters
        cell_to_idx = {cell: idx for idx, cell in enumerate(main_metadata['cells'])}
        time_to_idx = {str(time): idx for idx, time in enumerate(main_metadata['timepoints'])}
        
        cell_to_bicluster = dict(zip(cell_bicluster_df['cell'], cell_bicluster_df['cluster']))
        time_to_bicluster = dict(zip(time_bicluster_df['time'], time_bicluster_df['cluster']))
        
        # Create matrix for original data and cluster mapping
        projection = np.mean(main_tensor, axis=2)  # Average over features
        
        # Create a cluster matrix to map each point to a specific cluster
        # Format: (cell_bicluster_id, time_bicluster_id, cell_tricluster_id, time_tricluster_id)
        # -1 means not assigned to any tricluster
        cluster_matrix = np.full((len(main_metadata['cells']), len(main_metadata['timepoints']), 4), -1, dtype=int)
        
        # First, fill in the bicluster assignments for all cells and times
        for c_idx, cell in enumerate(main_metadata['cells']):
            if cell in cell_to_bicluster:
                cluster_matrix[c_idx, :, 0] = cell_to_bicluster[cell]
            
        for t_idx, time in enumerate(main_metadata['timepoints']):
            if str(time) in time_to_bicluster:
                cluster_matrix[:, t_idx, 1] = time_to_bicluster[str(time)]
        
        # Track how many unique triclusters we have
        tricluster_combos = set()
        
        # Now populate the tricluster assignments
        for key, result in all_results.items():
            if not result or 'bicluster_ids' not in result:
                continue
                
            cell_bic_id, time_bic_id = result['bicluster_ids']
            
            if 'cell_labels' not in result or 'time_labels' not in result or 'metadata' not in result:
                continue
                
            metadata = result['metadata']
            cell_labels = result['cell_labels'] 
            time_labels = result['time_labels']
            
            if not metadata or 'cells' not in metadata or 'timepoints' not in metadata:
                continue
                
            sub_cells = metadata['cells']
            sub_times = metadata['timepoints']
            
            # Check array lengths and truncate if needed
            min_cell_len = min(len(sub_cells), len(cell_labels))
            min_time_len = min(len(sub_times), len(time_labels))
            
            # Map each cell and time in this subtensor to its tricluster
            for c_idx in range(min_cell_len):
                cell = sub_cells[c_idx]
                cell_tri_id = cell_labels[c_idx]
                
                if cell not in cell_to_idx:
                    continue
                    
                main_c_idx = cell_to_idx[cell]
                
                for t_idx in range(min_time_len):
                    time = sub_times[t_idx]
                    time_tri_id = time_labels[t_idx]
                    
                    if str(time) not in time_to_idx:
                        continue
                        
                    main_t_idx = time_to_idx[str(time)]
                    
                    # Only assign if this point belongs to the correct bicluster
                    if (cluster_matrix[main_c_idx, main_t_idx, 0] == cell_bic_id and 
                        cluster_matrix[main_c_idx, main_t_idx, 1] == time_bic_id):
                        
                        # Assign tricluster IDs
                        cluster_matrix[main_c_idx, main_t_idx, 2] = cell_tri_id
                        cluster_matrix[main_c_idx, main_t_idx, 3] = time_tri_id
                        
                        # Record this unique combination
                        tricluster_combos.add((cell_bic_id, time_bic_id, cell_tri_id, time_tri_id))
        
        # Sort cells and times by bicluster
        sorted_cell_df = cell_bicluster_df.sort_values(['cluster', 'cell'])
        sorted_time_df = time_bicluster_df.sort_values(['cluster', 'time'])
        
        # Create sorted indices
        cell_indices = [cell_to_idx[cell] for cell in sorted_cell_df['cell'] if cell in cell_to_idx]
        time_indices = [time_to_idx[str(time)] for time in sorted_time_df['time'] if str(time) in time_to_idx]
        
        # Reorder matrices
        reordered_projection = projection[np.ix_(cell_indices, time_indices)]
        reordered_clusters = cluster_matrix[np.ix_(cell_indices, time_indices)]
        
        # Get bicluster boundaries
        cell_sizes = sorted_cell_df['cluster'].value_counts().sort_index()
        time_sizes = sorted_time_df['cluster'].value_counts().sort_index()
        
        cell_boundaries = np.cumsum(cell_sizes.values)[:-1]
        time_boundaries = np.cumsum(time_sizes.values)[:-1]
        
        # Count unique tricluster combinations
        n_tricluster_combos = len(tricluster_combos)
        logger.info(f"Found {n_tricluster_combos} unique tricluster combinations")
        
        # Set up color mapping - need a colormap with enough distinct colors
        if n_tricluster_combos <= 10:
            cmap_name = 'tab10'
        elif n_tricluster_combos <= 20:
            cmap_name = 'tab20'
        else:
            cmap_name = 'viridis'
            
        # Create color mapping
        cmap = plt.cm.get_cmap(cmap_name, max(n_tricluster_combos, 1))
        
        # Convert set to sorted list for consistent mapping
        sorted_combos = sorted(tricluster_combos)
        combo_to_idx = {combo: idx for idx, combo in enumerate(sorted_combos)}
        
        # Create a display matrix for the visualization
        display_matrix = np.full((len(cell_indices), len(time_indices)), -1.0, dtype=float)
        
        # Fill the display matrix with cluster indices
        for i, c_idx in enumerate(cell_indices):
            for j, t_idx in enumerate(time_indices):
                cell_bic = reordered_clusters[i, j, 0]
                time_bic = reordered_clusters[i, j, 1]
                cell_tri = reordered_clusters[i, j, 2]
                time_tri = reordered_clusters[i, j, 3]
                
                # Only fill in points that have a complete tricluster assignment
                if cell_bic >= 0 and time_bic >= 0 and cell_tri >= 0 and time_tri >= 0:
                    combo = (cell_bic, time_bic, cell_tri, time_tri)
                    if combo in combo_to_idx:
                        display_matrix[i, j] = combo_to_idx[combo]
        
        # Create masked array for visualization
        masked_display = np.ma.masked_where(display_matrix < 0, display_matrix)
        
        # Plot the original data and tricluster assignments
        plt.figure(figsize=(22, 12))
        
        # Plot original data
        plt.subplot(121)
        plt.imshow(reordered_projection, cmap='viridis', aspect='auto')
        plt.colorbar(label='Mean Feature Value')
        plt.title('Original Data\n(Cell-Time plane, reordered by biclusters)')
        
        # Add bicluster boundaries
        for x in time_boundaries:
            plt.axvline(x=x, color='white', linestyle='--', linewidth=1)
        
        for y in cell_boundaries:
            plt.axhline(y=y, color='white', linestyle='--', linewidth=1)
        
        # Add labels for bicluster regions
        # Add labels in the cell axis
        for i, (start, size) in enumerate(zip([0] + cell_boundaries.tolist(), cell_sizes)):
            mid_point = start + size / 2
            plt.text(-5, mid_point, f'C{i}', color='white', ha='center', va='center', 
                    bbox=dict(facecolor='black', alpha=0.5, boxstyle='round'))
            
        # Add labels in the time axis
        for i, (start, size) in enumerate(zip([0] + time_boundaries.tolist(), time_sizes)):
            mid_point = start + size / 2
            plt.text(mid_point, -5, f'T{i}', color='white', ha='center', va='center',
                   bbox=dict(facecolor='black', alpha=0.5, boxstyle='round'))
        
        # Set ticks
        x_tick_freq = max(1, len(time_indices) // 10)
        x_ticks = np.arange(0, len(time_indices), x_tick_freq)
        x_labels = [sorted_time_df['time'].iloc[i] if i < len(sorted_time_df) else '' for i in x_ticks]
        plt.xticks(x_ticks, x_labels, rotation=45)
        
        y_tick_freq = max(1, len(cell_indices) // 15)
        y_ticks = np.arange(0, len(cell_indices), y_tick_freq)
        y_labels = [sorted_cell_df['cell'].iloc[i] if i < len(sorted_cell_df) else '' for i in y_ticks]
        plt.yticks(y_ticks, y_labels)
        
        # Plot tricluster assignments
        plt.subplot(122)
        im = plt.imshow(masked_display, cmap=cmap, aspect='auto', interpolation='nearest')
        plt.title('Embedded Tricluster Assignments\n(Two-level Clustering)')
        
        # Add bicluster boundaries
        for x in time_boundaries:
            plt.axvline(x=x, color='white', linestyle='--', linewidth=1)
        
        for y in cell_boundaries:
            plt.axhline(y=y, color='white', linestyle='--', linewidth=1)
        
        # Add labels for bicluster regions
        # Add labels in the cell axis
        for i, (start, size) in enumerate(zip([0] + cell_boundaries.tolist(), cell_sizes)):
            mid_point = start + size / 2
            plt.text(-5, mid_point, f'C{i}', color='white', ha='center', va='center', 
                    bbox=dict(facecolor='black', alpha=0.5, boxstyle='round'))
            
        # Add labels in the time axis
        for i, (start, size) in enumerate(zip([0] + time_boundaries.tolist(), time_sizes)):
            mid_point = start + size / 2
            plt.text(mid_point, -5, f'T{i}', color='white', ha='center', va='center',
                   bbox=dict(facecolor='black', alpha=0.5, boxstyle='round'))
        
        # Set ticks
        plt.xticks(x_ticks, x_labels, rotation=45)
        plt.yticks(y_ticks, y_labels)
        
        # Save the main visualization
        plt.tight_layout()
        main_plot_path = output_dir / "embedded_tricluster_view.png"
        plt.savefig(main_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create a legend for the tricluster mapping
        legend_height = max(8, min(20, (n_tricluster_combos + 9) // 10))
        plt.figure(figsize=(12, legend_height))
        
        # Create patches for legend
        from matplotlib.patches import Patch
        legend_elements = []
        
        for combo, idx in combo_to_idx.items():
            cell_bic, time_bic, cell_tri, time_tri = combo
            color = cmap(idx)
            label = f"C{cell_bic}-T{time_bic}: Sub(C{cell_tri}-T{time_tri})"
            legend_elements.append(Patch(facecolor=color, label=label))
        
        # Sort for consistency
        legend_elements.sort(key=lambda x: x.get_label())
        
        # Create legend
        plt.axis('off')
        plt.legend(handles=legend_elements, loc='center', ncol=3, 
                   fontsize='small', title="Tricluster Legend")
        
        # Save legend
        legend_path = output_dir / "embedded_tricluster_legend.png"
        plt.savefig(legend_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Calculate statistics
        assigned_points = (display_matrix >= 0).sum()
        total_points = display_matrix.size
        coverage = assigned_points / total_points * 100
        
        logger.info(f"Embedded visualization saved to {main_plot_path}")
        logger.info(f"Legend saved to {legend_path}")
        logger.info(f"Assigned {assigned_points} out of {total_points} points, coverage: {coverage:.2f}%")
        
        # Create an alternate view that's more similar to cocluster_timeline.py
        # This will use a matrix where each bicluster region is shown in its own subplot
        
        plt.figure(figsize=(20, 15))
        plt.suptitle("Tricluster Assignments by Bicluster Region", fontsize=16)
        
        subplot_idx = 1
        max_subplots = n_cell_biclusters * n_time_biclusters
        
        # Determine subplot grid size
        grid_size = int(np.ceil(np.sqrt(max_subplots)))
        
        # Create a subplot for each bicluster region
        cell_boundaries = [0] + cell_boundaries.tolist() + [len(cell_indices)]
        time_boundaries = [0] + time_boundaries.tolist() + [len(time_indices)]
        
        for cell_bic in range(n_cell_biclusters):
            cell_start = cell_boundaries[cell_bic]
            cell_end = cell_boundaries[cell_bic + 1]
            
            for time_bic in range(n_time_biclusters):
                time_start = time_boundaries[time_bic]
                time_end = time_boundaries[time_bic + 1]
                
                # Skip if we have too many subplots
                if subplot_idx > grid_size * grid_size:
                    logger.warning(f"Too many bicluster regions, showing only the first {grid_size * grid_size}")
                    break
                
                # Extract region
                region = masked_display[cell_start:cell_end, time_start:time_end]
                
                # Skip if region is empty
                if np.all(region.mask):
                    logger.info(f"Skipping empty region C{cell_bic}-T{time_bic}")
                    continue
                
                # Create subplot
                plt.subplot(grid_size, grid_size, subplot_idx)
                plt.imshow(region, cmap=cmap, aspect='auto', interpolation='nearest')
                plt.title(f"C{cell_bic}-T{time_bic}")
                
                # Minimal ticks
                plt.xticks([])
                plt.yticks([])
                
                subplot_idx += 1
        
        # Save the subplots view
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave room for the suptitle
        subplots_path = output_dir / "tricluster_by_region.png"
        plt.savefig(subplots_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Region-based visualization saved to {subplots_path}")
        
    except Exception as e:
        logger.error(f"Error in embedded tricluster visualization: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

def main():
    """Main execution logic with improved error handling and parallel processing."""
    try:
        # Validate paths
        validate_paths()

        # Configuration
        N_BICLUSTERS = 9
        N_SUB_CELL_CLUSTERS = 3
        N_SUB_TIME_CLUSTERS = 2
        N_SUB_FEATURE_CLUSTERS = 3

        # Load and process timeline data
        timeline_data = load_timeline_data()
        if timeline_data is None:
            raise DataValidationError("Failed to load timeline data")

        # Perform bi-clustering
        cell_bicluster_df, time_bicluster_df = perform_biclustering(timeline_data, N_BICLUSTERS)
        if cell_bicluster_df is None or time_bicluster_df is None:
            raise DataValidationError("Bi-clustering failed")

        # Load main tensor and metadata
        main_tensor = np.load(OUTPUT_TENSOR_PATH)
        with open(OUTPUT_METADATA_PATH, 'r') as f:
            main_metadata = json.load(f)

        # Prepare parallel processing tasks
        tasks = []
        cell_name_to_idx = {name: idx for idx, name in enumerate(main_metadata['cells'])}
        time_point_to_idx = {str(point): idx for idx, point in enumerate(main_metadata['timepoints'])}

        for cell_clust_id in range(cell_bicluster_df['cluster'].nunique()):
            for time_clust_id in range(time_bicluster_df['cluster'].nunique()):
                # Get cells and times for this bi-cluster
                cells = cell_bicluster_df[cell_bicluster_df['cluster'] == cell_clust_id]['cell'].tolist()
                times = time_bicluster_df[time_bicluster_df['cluster'] == time_clust_id]['time'].tolist()

                # Get indices
                cell_indices = [cell_name_to_idx[cell] for cell in cells if cell in cell_name_to_idx]
                time_indices = [time_point_to_idx[f"{int(time):03d}"] for time in times if f"{int(time):03d}" in time_point_to_idx]

                if not cell_indices or not time_indices:
                    continue

                # Extract sub-tensor
                sub_tensor = main_tensor[np.ix_(cell_indices, time_indices, range(main_tensor.shape[2]))]

                # Prepare task
                task = {
                    'sub_tensor': sub_tensor,
                    'sub_metadata': {
                        'cells': [main_metadata['cells'][i] for i in cell_indices],
                        'timepoints': [main_metadata['timepoints'][i] for i in time_indices],
                        'features': main_metadata['features']
                    },
                    'n_cell_clusters': N_SUB_CELL_CLUSTERS,
                    'n_time_clusters': N_SUB_TIME_CLUSTERS,
                    'n_feature_clusters': N_SUB_FEATURE_CLUSTERS,
                    'bicluster_ids': (cell_clust_id, time_clust_id)
                }
                tasks.append(task)

        # Process sub-tensors in parallel
        results = {}
        with ProcessPoolExecutor() as executor:
            future_to_task = {executor.submit(process_subtensor, task): task for task in tasks}

            for future in tqdm(as_completed(future_to_task), total=len(tasks), desc="Processing sub-tensors"):
                task = future_to_task[future]
                try:
                    if result := future.result():
                        cell_id, time_id = result['bicluster_ids']
                        key = f"cell_{cell_id}_time_{time_id}"
                        results[key] = result

                        # Save individual result
                        output_path = OUTPUT_SUBCLUSTER_DIR / f"subcluster_C{cell_id}_T{time_id}_results.json"
                        with open(output_path, 'w') as f:
                            json.dump(result, f, indent=4)

                except Exception as e:
                    logger.error(f"Error processing task {task['bicluster_ids']}: {str(e)}")

        # Save aggregated results
        with open(OUTPUT_SUBCLUSTER_DIR / "all_subcluster_results.json", 'w') as f:
            json.dump(results, f, indent=4)

        # After processing results, add visualization
        OUTPUT_VIS_DIR = OUTPUT_SUBCLUSTER_DIR / "visualizations"
        visualize_coclusters(
            main_tensor=main_tensor,
            cell_bicluster_df=cell_bicluster_df,
            time_bicluster_df=time_bicluster_df,
            main_metadata=main_metadata,
            output_dir=OUTPUT_VIS_DIR
        )


        logger.info("Processing and visualization complete")

    except Exception as e:
        logger.error(f"Fatal error in main execution: {str(e)}")
        raise

def test_single_subtensor():
    """Test function to process a single sub-tensor without parallelization."""
    try:
        # Validate paths
        validate_paths()

        # Configuration
        N_BICLUSTERS = 9
        N_SUB_CELL_CLUSTERS = 3
        N_SUB_TIME_CLUSTERS = 2
        N_SUB_FEATURE_CLUSTERS = 3

        # Try to load existing bicluster results
        cell_clusters_path = OUTPUT_SUBCLUSTER_DIR / "cell_clusters.csv"
        time_clusters_path = OUTPUT_SUBCLUSTER_DIR / "time_clusters.csv"

        if cell_clusters_path.exists() and time_clusters_path.exists():
            logger.info("Loading existing bicluster results...")
            cell_bicluster_df = pd.read_csv(cell_clusters_path)
            time_bicluster_df = pd.read_csv(time_clusters_path)
        else:
            logger.info("Computing bicluster results...")
            # Load and process timeline data
            timeline_data = load_timeline_data()
            if timeline_data is None:
                raise DataValidationError("Failed to load timeline data")

            # Perform bi-clustering
            cell_bicluster_df, time_bicluster_df = perform_biclustering(timeline_data, N_BICLUSTERS)
            if cell_bicluster_df is None or time_bicluster_df is None:
                raise DataValidationError("Bi-clustering failed")

            # Save bicluster results for future use
            cell_bicluster_df.to_csv(cell_clusters_path, index=False)
            time_bicluster_df.to_csv(time_clusters_path, index=False)

        # Load main tensor and metadata
        main_tensor = np.load(OUTPUT_TENSOR_PATH)
        with open(OUTPUT_METADATA_PATH, 'r') as f:
            main_metadata = json.load(f)

        # Select a specific bicluster to test (e.g., cell cluster 0, time cluster 0)
        cell_clust_id = 0
        time_clust_id = 0

        # Create name to index mappings
        cell_name_to_idx = {name: idx for idx, name in enumerate(main_metadata['cells'])}
        time_point_to_idx = {str(point): idx for idx, point in enumerate(main_metadata['timepoints'])}

        # Get cells and times for this bi-cluster
        cells = cell_bicluster_df[cell_bicluster_df['cluster'] == cell_clust_id]['cell'].tolist()
        times = time_bicluster_df[time_bicluster_df['cluster'] == time_clust_id]['time'].tolist()

        # Get indices
        cell_indices = [cell_name_to_idx[cell] for cell in cells if cell in cell_name_to_idx]
        time_indices = [time_point_to_idx[f"{int(time):03d}"] for time in times if f"{int(time):03d}" in time_point_to_idx]

        if not cell_indices or not time_indices:
            raise ValueError("No valid indices found for the selected bicluster")

        # Extract sub-tensor
        sub_tensor = main_tensor[np.ix_(cell_indices, time_indices, range(main_tensor.shape[2]))]

        # Prepare task
        task = {
            'sub_tensor': sub_tensor,
            'sub_metadata': {
                'cells': [main_metadata['cells'][i] for i in cell_indices],
                'timepoints': [main_metadata['timepoints'][i] for i in time_indices],
                'features': main_metadata['features']
            },
            'n_cell_clusters': N_SUB_CELL_CLUSTERS,
            'n_time_clusters': N_SUB_TIME_CLUSTERS,
            'n_feature_clusters': N_SUB_FEATURE_CLUSTERS,
            'bicluster_ids': (cell_clust_id, time_clust_id)
        }

        # Process sub-tensor
        result = process_subtensor(task)

        if result:
            # Save result
            output_path = OUTPUT_SUBCLUSTER_DIR / f"test_subcluster_C{cell_clust_id}_T{time_clust_id}_results.json"
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=4)
            logger.info(f"Test result saved to {output_path}")
        else:
            logger.error("Failed to process sub-tensor")

    except Exception as e:
        logger.error(f"Error in test function: {str(e)}")
        raise

if __name__ == "__main__":
    # main()
    # test_single_subtensor()

    # Add tricluster visualization for only specific plots
    OUTPUT_TRI_VIS_DIR = OUTPUT_SUBCLUSTER_DIR / "tricluster_visualizations"
    visualize_triclusters(
        results_path=OUTPUT_SUBCLUSTER_DIR / "all_subcluster_results.json",
        output_dir=OUTPUT_TRI_VIS_DIR,
    )
    
    # Skip the embedded and combined visualizations
    logger.info("Skipping combined and embedded visualizations as requested")

# %%
