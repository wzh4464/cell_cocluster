import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralCoclustering
from sklearn.metrics import consensus_score
import pandas as pd
import seaborn as sns

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

def load_timeline_data(data_file='cell_timeline_data.txt'):
    """Load the timeline data from the text file."""
    # Read the data
    df = pd.read_csv(data_file, sep='\t', index_col=0)
    return df

def perform_coclustering(data, n_clusters=9):
    """Perform co-clustering on the data matrix."""
    # Convert to numpy array
    matrix = data.values
    
    # Create and fit the co-clustering model
    model = SpectralCoclustering(n_clusters=n_clusters, random_state=0)
    model.fit(matrix)
    
    # Get row and column labels
    row_labels = model.row_labels_
    col_labels = model.column_labels_
    
    # Get the original row and column names
    row_names = data.index
    col_names = data.columns.astype(int)  # Convert to integers for proper sorting
    
    # Create a DataFrame with the clustering results
    cluster_df = pd.DataFrame({
        'cell': row_names,
        'cluster': row_labels
    })
    
    time_df = pd.DataFrame({
        'time': col_names,
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

def main():
    # Load the data
    print("Loading timeline data...")
    data = load_timeline_data()
    
    print("Loading fate data...")
    fate_df = load_fate_data()
    
    print("Loading name dictionary...")
    name_dict = load_name_dictionary()
    
    # Perform co-clustering
    print("Performing co-clustering...")
    reordered_matrix, cluster_df, time_df = perform_coclustering(data)
    
    # Plot the results
    print("Plotting results...")
    plot_coclustered_matrix(reordered_matrix, cluster_df, time_df)
    
    # Analyze clusters
    print("Analyzing clusters...")
    analyze_clusters(reordered_matrix, cluster_df, time_df, fate_df, name_dict)
    
    print("\nDetailed cluster information has been saved to cluster_*_details.txt files")

if __name__ == '__main__':
    main() 