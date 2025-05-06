import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

def load_time_points(directory):
    """Load all time points from a directory."""
    files = [f for f in os.listdir(directory) if f.endswith('.npy')]
    if not files:
        return None
        
    time_points = []
    
    for file in files:
        try:
            # Extract time point from filename
            time_point = int(file.split('_')[1])
            time_points.append(time_point)
        except Exception as e:
            continue
    
    return sorted(time_points)

def create_cell_timeline_matrix(base_dir='DATA/spharm', output_file='cell_timeline.png'):
    """Create a matrix showing cell presence at different time points."""
    # Get all lineage directories
    lineage_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    lineage_dirs.sort()  # Sort alphabetically for consistent ordering
    
    if not lineage_dirs:
        print("Error: No lineage directories found!")
        return
    
    # Load time points for all lineages
    all_time_points = []
    valid_lineages = []
    
    print("Loading time points for all lineages...")
    for lineage_dir in tqdm(lineage_dirs):
        full_dir = os.path.join(base_dir, lineage_dir)
        time_points = load_time_points(full_dir)
        
        if time_points is not None:
            all_time_points.extend(time_points)
            valid_lineages.append(lineage_dir)
    
    if not valid_lineages:
        print("Error: No valid lineages found!")
        return
    
    # Get all unique time points
    all_unique_time_points = sorted(list(set(all_time_points)))
    
    print(f"Found {len(all_unique_time_points)} unique time points")
    print(f"Found {len(valid_lineages)} lineages")
    
    # Create presence matrix
    presence_matrix = np.zeros((len(valid_lineages), len(all_unique_time_points)), dtype=bool)
    
    # Fill presence matrix
    for i, lineage_dir in enumerate(valid_lineages):
        full_dir = os.path.join(base_dir, lineage_dir)
        time_points = load_time_points(full_dir)
        if time_points is not None:
            for time_point in time_points:
                j = all_unique_time_points.index(time_point)
                presence_matrix[i, j] = True
    
    # Create figure
    plt.figure(figsize=(20, 10))
    
    # Plot matrix
    plt.imshow(presence_matrix, aspect='auto', cmap='binary')
    
    # Set labels
    plt.xlabel('Time Points')
    plt.ylabel('Cell Lineages')
    
    # Set x-axis ticks (show every 10th time point)
    x_ticks = np.arange(0, len(all_unique_time_points), 10)
    plt.xticks(x_ticks, [all_unique_time_points[i] for i in x_ticks], rotation=45)
    
    # Set y-axis ticks (show every 10th lineage)
    y_ticks = np.arange(0, len(valid_lineages), 10)
    plt.yticks(y_ticks, [valid_lineages[i] for i in y_ticks])
    
    # Add colorbar
    plt.colorbar(label='Cell Presence')
    
    # Add title
    plt.title('Cell Presence Timeline')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Timeline matrix saved to {output_file}")
    
    # Also save the data to a text file
    with open('cell_timeline_data.txt', 'w') as f:
        # Write header
        f.write('Lineage\t' + '\t'.join(map(str, all_unique_time_points)) + '\n')
        # Write data
        for i, lineage in enumerate(valid_lineages):
            f.write(f"{lineage}\t" + '\t'.join(['1' if x else '0' for x in presence_matrix[i]]) + '\n')
    print("Data saved to cell_timeline_data.txt")

if __name__ == '__main__':
    create_cell_timeline_matrix() 