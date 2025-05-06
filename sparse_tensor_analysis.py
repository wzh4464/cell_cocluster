###
# File: ./sparse_tensor_analysis.py
# Created Date: Thursday, April 24th 2025
# Author: Zihan
# -----
# Last Modified: Thursday, 24th April 2025 10:58:32 pm
# Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
# -----
# HISTORY:
# Date      		By   	Comments
# ----------		------	---------------------------------------------------------
###

# %%
def analyze_tensor_sparsity(sparse_tensor, feature_names=None, plot_dir='plots'):
    """
    Analyze the sparsity pattern and value distribution of a sparse tensor.

    Parameters:
        sparse_tensor: COO format sparse tensor (coords, values, shape)
        feature_names: List of feature names (optional)
        plot_dir: Directory to save plots

    Returns:
        sparsity_stats: Dictionary containing sparsity statistics
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from collections import defaultdict

    coords, values, shape = sparse_tensor
    n_cells, n_timepoints, n_features = shape

    # Create plots directory if it doesn't exist
    os.makedirs(plot_dir, exist_ok=True)

    # 1. Overall sparsity analysis
    total_elements = n_cells * n_timepoints * n_features
    non_zero_elements = len(values)
    sparsity = 1 - (non_zero_elements / total_elements)

    # 2. Analyze sparsity by mode
    cell_sparsity = defaultdict(int)
    timepoint_sparsity = defaultdict(int)
    feature_sparsity = defaultdict(int)

    for coord in coords.T:
        cell_idx, time_idx, feature_idx = coord
        cell_sparsity[cell_idx] += 1
        timepoint_sparsity[time_idx] += 1
        feature_sparsity[feature_idx] += 1

    # 3. Value distribution analysis
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(values, bins=50)
    plt.title('Value Distribution')
    plt.xlabel('Value')
    plt.ylabel('Count')

    plt.subplot(1, 2, 2)
    sns.boxplot(x=values)
    plt.title('Value Box Plot')
    plt.xlabel('Value')

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'value_distribution.png'))
    plt.close()

    # 4. Sparsity patterns by mode
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.bar(range(n_cells), [cell_sparsity[i] for i in range(n_cells)])
    plt.title('Sparsity by Cell')
    plt.xlabel('Cell Index')
    plt.ylabel('Non-zero Entries')

    plt.subplot(1, 3, 2)
    plt.bar(range(n_timepoints), [timepoint_sparsity[i] for i in range(n_timepoints)])
    plt.title('Sparsity by Timepoint')
    plt.xlabel('Timepoint Index')
    plt.ylabel('Non-zero Entries')

    plt.subplot(1, 3, 3)
    plt.bar(range(n_features), [feature_sparsity[i] for i in range(n_features)])
    plt.title('Sparsity by Feature')
    plt.xlabel('Feature Index')
    plt.ylabel('Non-zero Entries')

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'sparsity_patterns.png'))
    plt.close()

    # 5. Feature correlation analysis (for non-sparse features)
    if feature_names is not None:
        # Create a dense matrix for features with sufficient data
        dense_features = []
        feature_indices = []
        for i in range(n_features):
            if feature_sparsity[i] > 0.1 * n_cells * n_timepoints:  # At least 10% non-zero
                feature_data = np.zeros((n_cells, n_timepoints))
                for coord, val in zip(coords.T, values):
                    if coord[2] == i:
                        feature_data[coord[0], coord[1]] = val
                dense_features.append(feature_data.flatten())
                feature_indices.append(i)

        if dense_features:
            feature_matrix = np.array(dense_features)
            correlation = np.corrcoef(feature_matrix)

            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation, 
                       xticklabels=[feature_names[i] for i in feature_indices],
                       yticklabels=[feature_names[i] for i in feature_indices],
                       cmap='coolwarm',
                       center=0)
            plt.title('Feature Correlation Matrix')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, 'feature_correlation.png'))
            plt.close()

    # 6. Timeline visualization
    plt.figure(figsize=(15, 8))
    
    # Define color scheme for both plots
    colors = plt.cm.tab20(np.linspace(0, 1, n_cells))
    
    for cell_idx in range(n_cells):
        cell_data = np.zeros(n_timepoints)
        for coord, val in zip(coords.T, values):
            if coord[0] == cell_idx:
                cell_data[coord[1]] = val
        
        plt.plot(range(n_timepoints), cell_data, 
                color=colors[cell_idx],
                alpha=0.7,
                label=f'Cell {cell_idx}')
    
    plt.title('Cell Feature Timeline')
    plt.xlabel('Timepoint')
    plt.ylabel('Feature Value')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'cell_timeline.png'))
    plt.close()

    # 7. Clustering visualization
    plt.figure(figsize=(12, 8))
    
    # Use the same color scheme as the timeline plot
    for cell_idx in range(n_cells):
        cell_data = np.zeros(n_features)
        for coord, val in zip(coords.T, values):
            if coord[0] == cell_idx:
                cell_data[coord[2]] = val
        
        plt.scatter(range(n_features), cell_data,
                   color=colors[cell_idx],
                   alpha=0.7,
                   label=f'Cell {cell_idx}')
    
    plt.title('Cell Feature Clustering')
    plt.xlabel('Feature Index')
    plt.ylabel('Feature Value')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'cell_clustering.png'))
    plt.close()

    # Compile statistics
    sparsity_stats = {
        "total_elements": total_elements,
        "non_zero_elements": non_zero_elements,
        "sparsity": sparsity,
        "cell_sparsity": dict(cell_sparsity),
        "timepoint_sparsity": dict(timepoint_sparsity),
        "feature_sparsity": dict(feature_sparsity),
        "value_stats": {
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
            "median": np.median(values),
        },
    }

    return sparsity_stats 

# %%

def create_cell_sparse_tensor(cell_data, time_data, feature_data, feature_names=None):
    """
    Create a sparse tensor for cell data with dimensions (cells × time × features).
    
    Parameters:
        cell_data: List or array of cell IDs
        time_data: List or array of time points
        feature_data: Dictionary or list of feature arrays for each cell and time point
                     Format: {(cell_id, time_point): feature_array}
        feature_names: List of feature names (optional)
        
    Returns:
        sparse_tensor: COO format sparse tensor (coords, values, shape)
        feature_names: List of feature names
    """
    import numpy as np
    from collections import defaultdict
    
    # Convert inputs to numpy arrays
    cell_data = np.array(cell_data)
    time_data = np.array(time_data)
    
    # Get unique cell IDs and time points
    unique_cells = np.unique(cell_data)
    unique_times = np.unique(time_data)
    
    # Create mappings for cell and time indices
    cell_map = {cell: idx for idx, cell in enumerate(unique_cells)}
    time_map = {time: idx for idx, time in enumerate(unique_times)}
    
    # Initialize lists for COO format
    coords = []
    values = []
    
    # Process feature data
    if isinstance(feature_data, dict):
        # If feature_data is a dictionary
        for (cell_id, time_point), features in feature_data.items():
            if cell_id in cell_map and time_point in time_map:
                cell_idx = cell_map[cell_id]
                time_idx = time_map[time_point]
                
                # Add non-zero features
                for feature_idx, value in enumerate(features):
                    if value != 0:  # Only store non-zero values
                        coords.append([cell_idx, time_idx, feature_idx])
                        values.append(value)
    
    elif isinstance(feature_data, list):
        # If feature_data is a list of arrays
        for i, cell_id in enumerate(cell_data):
            for j, time_point in enumerate(time_data):
                features = feature_data[i][j]
                cell_idx = cell_map[cell_id]
                time_idx = time_map[time_point]
                
                # Add non-zero features
                for feature_idx, value in enumerate(features):
                    if value != 0:  # Only store non-zero values
                        coords.append([cell_idx, time_idx, feature_idx])
                        values.append(value)
    
    # Convert to numpy arrays
    coords = np.array(coords).T  # Transpose for COO format
    values = np.array(values)
    
    # Determine shape
    n_cells = len(unique_cells)
    n_times = len(unique_times)
    n_features = len(feature_names) if feature_names is not None else max([c[2] for c in coords.T]) + 1
    
    shape = (n_cells, n_times, n_features)
    
    return (coords, values, shape), feature_names

def example_usage():
    """
    Example usage of create_cell_sparse_tensor
    """
    # Example data
    cell_ids = [1, 1, 2, 2, 3, 3]
    time_points = [0, 1, 0, 1, 0, 1]
    
    # Example feature data (cell_id, time_point): features
    feature_data = {
        (1, 0): [1.0, 0.0, 2.0],
        (1, 1): [0.0, 3.0, 0.0],
        (2, 0): [4.0, 0.0, 0.0],
        (2, 1): [0.0, 5.0, 6.0],
        (3, 0): [7.0, 0.0, 8.0],
        (3, 1): [0.0, 9.0, 0.0]
    }
    
    feature_names = ['feature1', 'feature2', 'feature3']
    
    # Create sparse tensor
    sparse_tensor, feature_names = create_cell_sparse_tensor(
        cell_ids, time_points, feature_data, feature_names
    )
    
    # Analyze sparsity
    sparsity_stats = analyze_tensor_sparsity(sparse_tensor, feature_names)
    
    print(f"Tensor shape: {sparse_tensor[2]}")
    print(f"Number of non-zero elements: {sparsity_stats['non_zero_elements']}")
    print(f"Sparsity: {sparsity_stats['sparsity']:.2%}")
    
    return sparse_tensor, feature_names

# %%

def create_cell_feature_tensor(data_dir):
    """
    Create a sparse tensor from cell features with dimensions (cells × time × features).
    
    Parameters:
        data_dir (str): Directory containing feature files
        
    Returns:
        sparse_tensor: COO format sparse tensor (coords, values, shape)
        feature_names: List of feature names
    """
    import numpy as np
    import os
    import glob
    import json
    from pathlib import Path
    
    # Define feature directories
    geo_dir = os.path.join(data_dir, 'geo_features')
    spharm_dir = os.path.join(data_dir, 'spharm')
    
    print(f"Looking for features in: {geo_dir}")
    print(f"Looking for spherical harmonics in: {spharm_dir}")
    
    # Get all unique cell IDs and timepoints
    all_cells = set()
    all_timepoints = set()
    
    # Get feature names from metadata (use the first cell's metadata)
    cell_dirs = [d for d in os.listdir(geo_dir) if os.path.isdir(os.path.join(geo_dir, d))]
    if not cell_dirs:
        raise ValueError(f"No cell directories found in {geo_dir}")
    
    print(f"Found {len(cell_dirs)} cell directories")
    print("First few cell directories:", cell_dirs[:5])
    
    # Get metadata from the first cell directory
    first_cell_dir = os.path.join(geo_dir, cell_dirs[0])
    print(f"Looking for metadata in: {first_cell_dir}")
    
    # List all files in the first cell directory
    print("Files in first cell directory:")
    for f in os.listdir(first_cell_dir):
        print(f"  {f}")
    
    # Try to find metadata file
    metadata_files = []
    for f in os.listdir(first_cell_dir):
        if f.endswith('_features_metadata.npy'):
            metadata_files.append(os.path.join(first_cell_dir, f))
    
    if not metadata_files:
        # If no metadata file found, use default feature names
        print("No metadata files found, using default feature names")
        geo_feature_names = [
            'volume', 'surface_area', 
            'centroid_x', 'centroid_y', 'centroid_z',
            'velocity_x', 'velocity_y', 'velocity_z',
            'acceleration_x', 'acceleration_y', 'acceleration_z'
        ]
    else:
        try:
            metadata = np.load(metadata_files[0], allow_pickle=True)
            real_data = metadata.item()  # 从0维数组中解包
            if isinstance(real_data, dict):
                geo_feature_names = real_data.get('feature_names', [])
                # 确保特征名称是字符串
                geo_feature_names = [str(name) for name in geo_feature_names]
            else:
                geo_feature_names = []
            print(f"Found {len(geo_feature_names)} feature names in metadata")
        except Exception as e:
            print(f"Error loading metadata: {str(e)}")
            geo_feature_names = [
                'volume', 'surface_area', 
                'centroid_x', 'centroid_y', 'centroid_z',
                'velocity_x', 'velocity_y', 'velocity_z',
                'acceleration_x', 'acceleration_y', 'acceleration_z'
            ]
    
    # Get spherical harmonics dimension
    spharm_files = glob.glob(os.path.join(spharm_dir, '*/*_l*.npy'))
    if spharm_files:
        print(f"Found {len(spharm_files)} spherical harmonics files")
        spharm_data = np.load(spharm_files[0], allow_pickle=True)
        spharm_size = spharm_data.size
        print(f"Spherical harmonics size: {spharm_size}")
    else:
        print("No spherical harmonics files found")
        spharm_size = 0
    
    # Scan data files to find dimensions
    for cell_dir in cell_dirs:
        try:
            # 保持细胞ID为浮点数
            cell_id = float(cell_dir.split('_')[1])  # Extract number from cell_XXX
            cell_path = os.path.join(geo_dir, cell_dir)
            
            # Get all feature files for this cell
            feature_files = glob.glob(os.path.join(cell_path, '*_features.npy'))
            for feature_file in feature_files:
                base = os.path.basename(feature_file)
                timepoint = int(base.split('_')[2])  # Extract timepoint from cell_XXX_YYY_features.npy
                
                all_cells.add(cell_id)
                all_timepoints.add(timepoint)
        except Exception as e:
            print(f"Error processing {cell_dir}: {str(e)}")
            continue
    
    print(f"Found {len(all_cells)} unique cells")
    print(f"Found {len(all_timepoints)} unique timepoints")
    
    # Create mappings
    cell_ids = sorted(all_cells)
    timepoints = sorted(all_timepoints)
    
    cell_map = {cell: idx for idx, cell in enumerate(cell_ids)}
    time_map = {time: idx for idx, time in enumerate(timepoints)}
    
    # Initialize lists for COO format
    coords = []
    values = []
    
    # Feature names
    feature_names = []
    if isinstance(geo_feature_names, list):
        feature_names.extend(geo_feature_names)
    # 确保球谐系数的名称也是字符串
    feature_names.extend([f"spharm_{i}" for i in range(spharm_size)])
    
    print(f"Total number of features: {len(feature_names)}")
    
    # Load geometric features
    for cell_id in cell_ids:
        cell_dir = os.path.join(geo_dir, f"cell_{cell_id}")  # 使用浮点数格式
        if not os.path.exists(cell_dir):
            print(f"Directory not found: {cell_dir}")
            continue
            
        for timepoint in timepoints:
            geo_file = os.path.join(cell_dir, f"cell_{cell_id}_{timepoint:03d}_features.npy")
            if os.path.exists(geo_file):
                try:
                    geo_data = np.load(geo_file, allow_pickle=True)
                    
                    # Convert object array to float array if needed
                    if geo_data.dtype == np.object_:
                        geo_data = geo_data.astype(np.float64)
                    
                    for i, val in enumerate(geo_data):
                        if val != 0:  # Only store non-zero values
                            coords.append([cell_map[cell_id], time_map[timepoint], i])
                            values.append(val)
                except Exception as e:
                    print(f"Error loading {geo_file}: {str(e)}")
                    continue
    
    # Load spherical harmonics features
    for cell_id in cell_ids:
        # Find the cell prefix (AB, ABa, etc.)
        cell_prefix = None
        for prefix in ['AB', 'ABa']:  # Add more prefixes if needed
            if os.path.exists(os.path.join(spharm_dir, prefix)):
                cell_prefix = prefix
                break
        
        if cell_prefix is None:
            print(f"No spherical harmonics prefix found for cell {cell_id}")
            continue
            
        for timepoint in timepoints:
            spharm_pattern = os.path.join(spharm_dir, cell_prefix, f"{cell_prefix}_{timepoint:03d}_l*.npy")
            spharm_files = glob.glob(spharm_pattern)
            
            if spharm_files:
                try:
                    spharm_data = np.load(spharm_files[0], allow_pickle=True)
                    
                    # Convert object array to float array if needed
                    if spharm_data.dtype == np.object_:
                        spharm_data = spharm_data.astype(np.float64)
                    
                    for i, val in enumerate(spharm_data.flatten()):
                        if val != 0:  # Only store non-zero values
                            feature_idx = len(geo_feature_names) + i
                            coords.append([cell_map[cell_id], time_map[timepoint], feature_idx])
                            values.append(val)
                except Exception as e:
                    print(f"Error loading {spharm_files[0]}: {str(e)}")
                    continue
    
    # Convert to numpy arrays
    coords = np.array(coords).T  # Transpose for COO format
    values = np.array(values)
    shape = (len(cell_ids), len(timepoints), len(feature_names))
    
    print(f"Created sparse tensor with shape {shape}")
    print(f"Number of non-zero elements: {len(values)}")
    
    return (coords, values, shape), feature_names

def analyze_cell_features(data_dir):
    """
    Analyze cell features and create sparse tensor representation.
    
    Parameters:
        data_dir (str): Directory containing feature files
        
    Returns:
        tuple: (sparse_tensor, feature_names, sparsity_stats)
    """
    # Create sparse tensor
    print("Creating sparse tensor from cell features...")
    sparse_tensor, feature_names = create_cell_feature_tensor(data_dir)
    
    # Analyze sparsity
    print("Analyzing tensor sparsity...")
    sparsity_stats = analyze_tensor_sparsity(sparse_tensor, feature_names)
    
    # Print summary
    print("\nFeature Analysis Summary:")
    print(f"Number of cells: {sparse_tensor[2][0]}")
    print(f"Number of timepoints: {sparse_tensor[2][1]}")
    print(f"Number of features: {sparse_tensor[2][2]}")
    print(f"Total non-zero elements: {sparsity_stats['non_zero_elements']}")
    print(f"Sparsity: {sparsity_stats['sparsity']:.2%}")
    
    return sparse_tensor, feature_names, sparsity_stats

# %%
# 使用正确的数据目录路径
data_dir = "DATA"  # 这是包含 geo_features 和 spharm 目录的根目录

try:
    # 创建并分析稀疏张量
    sparse_tensor, feature_names, sparsity_stats = analyze_cell_features(data_dir)
    
    # 查看特征名称
    print("\nFeature names:")
    for i, name in enumerate(feature_names):
        print(f"{i}: {name}")
    
    # 查看张量形状
    print("\nTensor shape (cells × timepoints × features):", sparse_tensor[2])
    
    # 查看稀疏性统计
    print("\nSparsity statistics:")
    print(f"Total elements: {sparsity_stats['total_elements']}")
    print(f"Non-zero elements: {sparsity_stats['non_zero_elements']}")
    print(f"Sparsity: {sparsity_stats['sparsity']:.2%}")
    
    # 查看每个维度的稀疏性
    print("\nSparsity by dimension:")
    print("Cells with most non-zero entries:")
    cell_sparsity = sorted(sparsity_stats['cell_sparsity'].items(), 
                         key=lambda x: x[1], reverse=True)[:5]
    for cell_idx, count in cell_sparsity:
        print(f"  Cell {cell_idx:03d}: {count} non-zero entries")
    
    print("\nTimepoints with most non-zero entries:")
    time_sparsity = sorted(sparsity_stats['timepoint_sparsity'].items(), 
                         key=lambda x: x[1], reverse=True)[:5]
    for time_idx, count in time_sparsity:
        print(f"  Timepoint {time_idx:03d}: {count} non-zero entries")
    
    print("\nFeatures with most non-zero entries:")
    feature_sparsity = sorted(sparsity_stats['feature_sparsity'].items(), 
                            key=lambda x: x[1], reverse=True)[:5]
    for feature_idx, count in feature_sparsity:
        print(f"  {feature_names[feature_idx]}: {count} non-zero entries")

except Exception as e:
    print(f"Error: {str(e)}")
    print("\nMake sure your data directory structure is correct:")
    print("DATA/")
    print("├── geo_features/")
    print("│   ├── cell_001/")
    print("│   │   ├── cell_001_001_features.npy")
    print("│   │   └── cell_001_001_features_metadata.json")
    print("└── spharm/")
    print("    ├── AB/")
    print("    │   └── AB_001_l15.npy")
    print("    └── ABa/")
    print("        └── ABa_001_l15.npy")

# %%
