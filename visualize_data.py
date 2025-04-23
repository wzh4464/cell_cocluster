import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from read_data import load_nifti_file, get_all_nifti_files

def normalize_coordinates(x, y, z):
    """Normalize coordinates to [0,1] range."""
    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y - y.min()) / (y.max() - y.min())
    z_norm = (z - z.min()) / (z.max() - z.min())
    return x_norm, y_norm, z_norm

def visualize_first_frame(data_dir):
    """Visualize the first frame of the 3D cell labels."""
    # Get the first NIfTI file
    nifti_files = get_all_nifti_files(data_dir)
    if not nifti_files:
        raise FileNotFoundError("No NIfTI files found in the data directory")
    
    first_file = nifti_files[0]
    volume = load_nifti_file(str(first_file))
    
    # Get unique labels (excluding background 0)
    unique_labels = np.unique(volume)[1:]
    
    # Create a figure with 3D axes
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Generate a color map for cells
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    # Plot each cell with a different color
    for idx, label in enumerate(unique_labels):
        # Get coordinates where this label appears
        x, y, z = np.where(volume == label)
        
        # Normalize coordinates
        x_norm, y_norm, z_norm = normalize_coordinates(x, y, z)
        
        # Plot the points for this cell
        ax.scatter(x_norm, y_norm, z_norm, 
                  color=colors[idx], 
                  alpha=0.6, 
                  label=f'Cell {int(label)}')
    
    # Set labels and title
    ax.set_xlabel('X (normalized)')
    ax.set_ylabel('Y (normalized)')
    ax.set_zlabel('Z (normalized)')
    ax.set_title('3D Cell Labels Visualization (First Frame)')
    
    # Add a legend
    ax.legend(bbox_to_anchor=(1.15, 1), loc='upper left')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Show the plot
    plt.savefig('first_frame_visualization.png')
    plt.show()

def main():
    data_dir = "data"
    visualize_first_frame(data_dir)

if __name__ == "__main__":
    main()
