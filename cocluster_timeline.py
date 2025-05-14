import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralCoclustering
from sklearn.metrics import consensus_score
import pandas as pd
import os
from scipy.special import logsumexp
from scipy.optimize import minimize
from sklearn.utils import check_random_state
from scipy.linalg import svd
from multiprocessing import Pool, cpu_count
from functools import partial

def load_name_dictionary(dict_file='DATA/name_dictionary.csv'):
    """Load the cell name dictionary."""
    name_dict = {}
    with open(dict_file, 'r') as f:
        for line in f:
            # Skip empty lines
            if not line.strip():
                continue
            parts = line.strip().split(',')
            if len(parts) == 2:
                try:
                    index, name = parts
                    # Skip if index is empty
                    if not index.strip():
                        continue
                    # Store both index->name and name->index mappings
                    name_dict[int(float(index))] = name
                    name_dict[name] = int(float(index))
                except (ValueError, IndexError):
                    print(f"Warning: Skipping invalid line: {line.strip()}")
                    continue
    return name_dict

def load_fate_data(fate_file='DATA/fate.csv'):
    """Load the cell fate data."""
    fate_df = pd.read_csv(fate_file, header=None, names=['cell', 'fate'])
    return fate_df

def load_timeline_data(data_file=os.path.join("logs", "cell_timeline_data.txt")):
    """Load the timeline data from the text file."""
    # Read the data
    df = pd.read_csv(data_file, sep='\t', index_col=0)
    return df

def update_row_labels(args):
    """并行更新行标签的辅助函数"""
    i, matrix, n_clusters, col_labels, block_probs = args
    log_probs = np.zeros(n_clusters)
    for k in range(n_clusters):
        for j in range(len(col_labels)):
            l = col_labels[j]
            p = block_probs[k, l]
            log_probs[k] += matrix[i, j] * np.log(p) + (1 - matrix[i, j]) * np.log(1 - p)
    return i, np.argmax(log_probs)

def update_col_labels(args):
    """并行更新列标签的辅助函数"""
    j, matrix, n_clusters, row_labels, block_probs = args
    log_probs = np.zeros(n_clusters)
    for l in range(n_clusters):
        for i in range(len(row_labels)):
            k = row_labels[i]
            p = block_probs[k, l]
            log_probs[l] += matrix[i, j] * np.log(p) + (1 - matrix[i, j]) * np.log(1 - p)
    return j, np.argmax(log_probs)

def update_block_probs(args):
    """并行更新块概率的辅助函数"""
    k, l, matrix, row_labels, col_labels = args
    mask = (row_labels == k)[:, None] & (col_labels == l)
    if np.sum(mask) > 0:
        return k, l, np.mean(matrix[mask])
    return k, l, 0.5

def perform_coclustering(data, n_clusters=9, random_state=0, n_jobs=None):
    """Perform co-clustering using Bernoulli-Latent Block Model with parallel processing.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The input data matrix
    n_clusters : int, default=9
        The number of biclusters to find
    random_state : int, default=0
        Random seed for reproducibility
    n_jobs : int, default=None
        Number of parallel jobs. If None, uses all available cores.
    """
    # Convert to numpy array
    matrix = data.values
    
    # Initialize parameters
    n_rows, n_cols = matrix.shape
    rng = check_random_state(random_state)
    
    # Initialize row and column cluster assignments
    row_labels = rng.randint(0, n_clusters, size=n_rows)
    col_labels = rng.randint(0, n_clusters, size=n_cols)
    
    # Initialize block probabilities
    block_probs = np.zeros((n_clusters, n_clusters))
    for i in range(n_clusters):
        for j in range(n_clusters):
            mask = (row_labels == i)[:, None] & (col_labels == j)
            if np.sum(mask) > 0:
                block_probs[i, j] = np.mean(matrix[mask])
            else:
                block_probs[i, j] = 0.5
    
    # Set number of parallel jobs
    if n_jobs is None:
        n_jobs = cpu_count()
    
    # EM algorithm
    max_iter = 100
    tol = 1e-6
    prev_log_likelihood = -np.inf
    
    with Pool(n_jobs) as pool:
        for iteration in range(max_iter):
            # E-step: Update row and column cluster assignments in parallel
            # Update row labels
            row_args = [(i, matrix, n_clusters, col_labels, block_probs) for i in range(n_rows)]
            row_updates = pool.map(update_row_labels, row_args)
            for i, label in row_updates:
                row_labels[i] = label
            
            # Update column labels
            col_args = [(j, matrix, n_clusters, row_labels, block_probs) for j in range(n_cols)]
            col_updates = pool.map(update_col_labels, col_args)
            for j, label in col_updates:
                col_labels[j] = label
            
            # M-step: Update block probabilities in parallel
            block_args = [(k, l, matrix, row_labels, col_labels) 
                         for k in range(n_clusters) for l in range(n_clusters)]
            block_updates = pool.map(update_block_probs, block_args)
            for k, l, prob in block_updates:
                block_probs[k, l] = prob
            
            # Calculate log-likelihood
            log_likelihood = 0
            for i in range(n_rows):
                for j in range(n_cols):
                    k = row_labels[i]
                    l = col_labels[j]
                    p = block_probs[k, l]
                    log_likelihood += matrix[i, j] * np.log(p) + (1 - matrix[i, j]) * np.log(1 - p)
            
            # Check convergence
            if abs(log_likelihood - prev_log_likelihood) < tol:
                break
            prev_log_likelihood = log_likelihood
    
    # Create DataFrames for the clustering results
    cluster_df = pd.DataFrame({
        'cell': data.index,
        'cluster': row_labels
    })
    
    time_df = pd.DataFrame({
        'time': data.columns,
        'cluster': col_labels
    })
    
    # Sort by cluster and then by name/time
    cluster_df = cluster_df.sort_values(['cluster', 'cell'])
    time_df = time_df.sort_values(['cluster', 'time'])
    
    # Reorder the matrix based on the sorted indices
    reordered_matrix = matrix[cluster_df.index]
    reordered_matrix = reordered_matrix[:, time_df.index]
    
    return reordered_matrix, cluster_df, time_df

def plot_coclustered_matrix(matrix, cluster_df, time_df, output_file='coclustered_timeline.png'):
    """Plot the co-clustered matrix."""
    plt.figure(figsize=(20, 10))
    
    # Define a color palette that matches the fate distribution plot
    colors = plt.cm.tab10(np.linspace(0, 1, 9))
    
    # Rasterize matrix with cluster colors via imshow
    n_clust = 9
    cmap = plt.cm.get_cmap('tab10', 9)
    row_lbls = cluster_df['cluster'].values
    col_lbls = time_df['cluster'].values
    # Build a matrix of cluster indices for points, mask non-points
    cluster_idx = np.full_like(matrix, fill_value=-1, dtype=int)
    for k in range(n_clust):
        mask = (row_lbls[:, None] == k) & (col_lbls[None, :] == k) & (matrix == 1)
        cluster_idx[mask] = k
    masked = np.ma.masked_where(cluster_idx < 0, cluster_idx)
    
    # Create a normalization object to map cluster indices to [0, 1]
    norm = plt.Normalize(vmin=0, vmax=8)
    plt.imshow(masked, cmap=cmap, interpolation='nearest', aspect='auto', norm=norm)
    plt.colorbar(ticks=np.arange(9), label='Cluster')
    
    # Add cluster boundaries
    cluster_sizes = cluster_df['cluster'].value_counts().sort_index()
    time_sizes = time_df['cluster'].value_counts().sort_index()
    
    # Add vertical lines for column clusters
    col_cumsum = np.cumsum(time_sizes)
    for x in col_cumsum[:-1]:
        plt.axvline(x=x, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Add horizontal lines for row clusters
    row_cumsum = np.cumsum(cluster_sizes)
    for y in row_cumsum[:-1]:
        plt.axhline(y=y, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Set labels
    plt.xlabel('Time Points')
    plt.ylabel('Cell Lineages')
    
    # Set x-axis ticks (show every 10th time point)
    x_ticks = np.arange(0, matrix.shape[1], 10)
    plt.xticks(x_ticks, [time_df['time'].iloc[i] for i in x_ticks], rotation=45)
    
    # Set y-axis ticks (show every 10th lineage)
    y_ticks = np.arange(0, matrix.shape[0], 10)
    plt.yticks(y_ticks, [cluster_df['cell'].iloc[i] for i in y_ticks])
    
    plt.title('Co-clustered Cell Timeline')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Co-clustered matrix plot saved to {output_file}")

def analyze_clusters(matrix, cluster_df, time_df, fate_df, name_dict):
    """Analyze and print information about the clusters."""
    n_clusters = len(cluster_df['cluster'].unique())
    
    print(f"\nFound {n_clusters} clusters")
    
    # Create a DataFrame to store fate distribution for each cluster
    fate_distribution = pd.DataFrame()
    
    # Save detailed cluster information to files
    for i in range(n_clusters):
        # Get cells in this cluster
        cluster_cells = cluster_df[cluster_df['cluster'] == i]['cell']
        # Get times in this cluster
        cluster_times = time_df[time_df['cluster'] == i]['time']
        
        # Calculate cluster density
        cluster_rows = matrix[cluster_df['cluster'] == i]
        cluster_cols = cluster_rows[:, time_df['cluster'] == i]
        density = np.mean(cluster_cols)
        
        # Get fate information for cells in this cluster
        cluster_fates = fate_df[fate_df['cell'].isin(cluster_cells)]
        fate_counts = cluster_fates['fate'].value_counts()
        
        # Store fate distribution
        fate_distribution[f'Cluster {i}'] = fate_counts
        
        print(f"\nCluster {i}:")
        print(f"Number of cells: {len(cluster_cells)}")
        print(f"Number of time points: {len(cluster_times)}")
        print(f"Density: {density:.2f}")
        print("\nFate distribution:")
        print(fate_counts)
        
        # Save cluster details to file
        with open(f'cluster_{i}_details.txt', 'w') as f:
            f.write(f"Cluster {i} Details\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Cells in this cluster:\n")
            f.write("-" * 20 + "\n")
            for cell in cluster_cells:
                # Get cell name (it's already the name, no need to convert)
                cell_name = cell
                # Check if cell exists in fate_df
                cell_fate = fate_df[fate_df['cell'] == cell]
                if not cell_fate.empty:
                    fate = cell_fate['fate'].iloc[0]
                    f.write(f"{cell_name} ({fate})\n")
                else:
                    f.write(f"{cell_name} (Fate unknown)\n")
            
            f.write("\nTime points in this cluster:\n")
            f.write("-" * 20 + "\n")
            for time in cluster_times:
                f.write(f"{time}\n")
            
            f.write(f"\nCluster density: {density:.2f}\n")
            
            f.write("\nFate distribution:\n")
            f.write("-" * 20 + "\n")
            for fate, count in fate_counts.items():
                f.write(f"{fate}: {count}\n")
            
            # Add information about cells with unknown fate
            unknown_fate_cells = len(cluster_cells) - len(cluster_fates)
            if unknown_fate_cells > 0:
                f.write(f"\nCells with unknown fate: {unknown_fate_cells}\n")
    
    # Plot fate distribution across clusters
    plt.figure(figsize=(12, 8))
    fate_distribution.plot(kind='bar', stacked=True)
    plt.title('Fate Distribution Across Clusters')
    plt.xlabel('Fate')
    plt.ylabel('Number of Cells')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('fate_distribution.png', dpi=300, bbox_inches='tight')
    print("\nFate distribution plot saved to fate_distribution.png")

def estimate_clusters_by_svd(data, threshold=0.95):
    """使用SVD秩估计来确定最佳的聚类数目。
    
    Parameters:
    -----------
    data : pandas.DataFrame
        输入数据矩阵
    threshold : float, default=0.95
        奇异值累积贡献率的阈值
        
    Returns:
    --------
    int
        估计的聚类数目
    """
    # 转换为numpy数组
    matrix = data.values
    
    # 计算SVD
    U, s, Vh = svd(matrix)
    
    # 计算奇异值的累积贡献率
    cumsum = np.cumsum(s) / np.sum(s)
    
    # 找到第一个超过阈值的索引
    n_clusters = np.argmax(cumsum >= threshold) + 1
    
    # 绘制奇异值分布图
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(s) + 1), s, 'b-', label='Singular Values')
    plt.plot(range(1, len(s) + 1), cumsum, 'r--', label='Cumulative Sum')
    plt.axvline(x=n_clusters, color='g', linestyle='--', label=f'Estimated Clusters: {n_clusters}')
    plt.axhline(y=threshold, color='k', linestyle=':', label=f'Threshold: {threshold}')
    plt.xlabel('Component Number')
    plt.ylabel('Value')
    plt.title('SVD Analysis for Cluster Number Estimation')
    plt.legend()
    plt.grid(True)
    plt.savefig('svd_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Estimated number of clusters: {n_clusters}")
    print(f"Singular values: {s[:n_clusters]}")
    print(f"Cumulative contribution: {cumsum[n_clusters-1]:.4f}")
    
    return n_clusters

def main():
    # Load the data
    print("Loading timeline data...")
    data = load_timeline_data()
    
    print("Loading fate data...")
    fate_df = load_fate_data()
    
    print("Loading name dictionary...")
    name_dict = load_name_dictionary()
    
    # Estimate number of clusters using SVD
    print("Estimating number of clusters using SVD...")
    n_clusters = 3
    random_state = 0  # 固定随机种子以确保结果可复现
    
    # Get number of available CPU cores
    n_jobs = cpu_count()
    print(f"Using {n_jobs} CPU cores for parallel processing")
    
    # Perform co-clustering
    print("Performing co-clustering...")
    reordered_matrix, cluster_df, time_df = perform_coclustering(
        data,
        n_clusters=n_clusters,
        random_state=random_state,
        n_jobs=n_jobs
    )
    
    # Plot the results
    print("Plotting results...")
    plot_coclustered_matrix(reordered_matrix, cluster_df, time_df)
    
    # Analyze clusters
    print("Analyzing clusters...")
    analyze_clusters(reordered_matrix, cluster_df, time_df, fate_df, name_dict)
    
    print("\nDetailed cluster information has been saved to cluster_*_details.txt files")

if __name__ == '__main__':
    main()
