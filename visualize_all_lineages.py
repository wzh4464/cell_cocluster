# %%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from scipy.special import sph_harm
import matplotlib.animation as animation
from tqdm import tqdm
from PIL import Image
import io

def load_coefficients(directory):
    """Load all coefficient files from a directory and sort by time point."""
    files = [f for f in os.listdir(directory) if f.endswith('.npy')]
    if not files:
        print(f"Warning: No .npy files found in {directory}")
        return None, None

    time_points = []
    coefficients = []

    for file in files:
        try:
            # Extract time point from filename
            time_point = int(file.split('_')[1])
            time_points.append(time_point)

            # Load coefficients
            coeffs = np.load(os.path.join(directory, file))
            coefficients.append(coeffs)
        except Exception as e:
            print(f"Warning: Error processing file {file}: {str(e)}")
            continue

    if not time_points:
        print(f"Warning: No valid time points found in {directory}")
        return None, None

    # Sort by time point
    sorted_indices = np.argsort(time_points)
    time_points = np.array(time_points)[sorted_indices]
    coefficients = np.array(coefficients)[sorted_indices]

    return time_points, coefficients

def reconstruct_shape(coefficients, theta, phi):
    """Reconstruct 3D shape from spherical harmonic coefficients."""
    l_max = 15  # Maximum degree of spherical harmonics
    shape = np.zeros_like(theta, dtype=complex)
    
    idx = 0
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            # Get coefficient for this l,m
            coeff = coefficients[idx]
            # Add contribution from this spherical harmonic
            shape += coeff * sph_harm(m, l, theta, phi)
            idx += 1
    
    return np.real(shape)

def create_all_lineages_animation(base_dir='DATA/spharm', output_file='all_lineages_animation.gif'):
    """Create animation showing all cell lineages in a grid layout."""
    # Get all lineage directories
    lineage_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    lineage_dirs.sort()  # Sort alphabetically for consistent ordering

    if not lineage_dirs:
        print("Error: No lineage directories found!")
        return

    # Load data for all lineages
    all_time_points = []
    all_coefficients = []
    valid_lineages = []

    print("Loading data for all lineages...")
    for lineage_dir in tqdm(lineage_dirs):
        full_dir = os.path.join(base_dir, lineage_dir)
        time_points, coefficients = load_coefficients(full_dir)

        if time_points is not None and coefficients is not None:
            all_time_points.append(time_points)
            all_coefficients.append(coefficients)
            valid_lineages.append(lineage_dir)

    if not valid_lineages:
        print("Error: No valid lineages found!")
        return

    # Get all unique time points
    all_unique_time_points = set()
    for tp in all_time_points:
        all_unique_time_points.update(tp)
    all_time_points_sorted = sorted(list(all_unique_time_points))

    print(f"Found {len(all_time_points_sorted)} unique time points")
    print(f"Visualizing {len(valid_lineages)} lineages")

    # Create grid layout
    n_lineages = len(valid_lineages)
    n_cols = int(np.ceil(np.sqrt(n_lineages)))
    n_rows = int(np.ceil(n_lineages / n_cols))

    # Create grid for visualization
    theta = np.linspace(0, np.pi, 20)  # Further reduced resolution for faster rendering
    phi = np.linspace(0, 2*np.pi, 20)
    theta, phi = np.meshgrid(theta, phi)

    # Convert to Cartesian coordinates
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    # Create figure with smaller size
    fig = plt.figure(figsize=(15, 15))

    # Store frames
    frames = []

    print("Generating frames...")
    for time_point in tqdm(all_time_points_sorted):
        plt.clf()

        for i, (lineage_dir, coefficients, lineage_time_points) in enumerate(zip(valid_lineages, all_coefficients, all_time_points)):
            ax = fig.add_subplot(n_rows, n_cols, i+1, projection='3d')

            # Check if this lineage has data for this time point
            if time_point in lineage_time_points:
                # Find the index of this time point in the lineage's time points
                time_idx = np.where(lineage_time_points == time_point)[0][0]

                try:
                    # Reconstruct shape
                    r = reconstruct_shape(coefficients[time_idx], theta, phi)

                    # Plot surface
                    ax.plot_surface(x*r, y*r, z*r, cmap='viridis')
                    ax.set_title(f'{lineage_dir}\nTime: {time_point}', fontsize=8)
                except Exception as e:
                    print(f"Warning: Error reconstructing shape for {lineage_dir} at time {time_point}: {str(e)}")
                    ax.set_title(f'{lineage_dir}\nError', fontsize=8)
            else:
                # If no data for this time point, show empty plot
                ax.set_title(f'{lineage_dir}\nNo data', fontsize=8)

            ax.set_xlim([-2, 2])
            ax.set_ylim([-2, 2])
            ax.set_zlim([-2, 2])
            ax.axis('off')  # Hide axes for cleaner visualization

        plt.suptitle(f'Time Point: {time_point}', fontsize=16)
        plt.tight_layout()

        # Save frame to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        frames.append(Image.open(buf))
        buf.close()

    # Save as GIF
    print("Saving animation...")
    frames[0].save(
        output_file,
        save_all=True,
        append_images=frames[1:],
        duration=200,  # 200ms per frame
        loop=0  # Infinite loop
    )
    print(f"Animation saved to {output_file}")
    plt.close()

if __name__ == '__main__':
    create_all_lineages_animation() 

# %%
