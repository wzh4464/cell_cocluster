# %%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from scipy.special import sph_harm
import matplotlib.animation as animation
from tqdm import tqdm

def load_coefficients(directory):
    """Load all coefficient files from a directory and sort by time point."""
    files = [f for f in os.listdir(directory) if f.endswith('.npy')]
    time_points = []
    coefficients = []
    
    for file in files:
        # Extract time point from filename
        time_point = int(file.split('_')[1])
        time_points.append(time_point)
        
        # Load coefficients
        coeffs = np.load(os.path.join(directory, file))
        coefficients.append(coeffs)
    
    # Sort by time point
    sorted_indices = np.argsort(time_points)
    time_points = np.array(time_points)[sorted_indices]
    coefficients = np.array(coefficients)[sorted_indices]
    
    return time_points, coefficients

def reconstruct_shape(coefficients, theta, phi):
    """Reconstruct 3D shape from spherical harmonic coefficients."""
    l_max = 15  # Maximum degree of spherical harmonics
    shape = np.zeros_like(theta, dtype=complex)
    
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            # Get coefficient for this l,m
            coeff = coefficients[l*(l+1) + m]
            # Add contribution from this spherical harmonic
            shape += coeff * sph_harm(m, l, theta, phi)
    
    return np.real(shape)

def create_animation(directory, output_file='lineage_animation.mp4'):
    """Create animation of cell shape evolution."""
    # Load coefficients
    time_points, coefficients = load_coefficients(directory)
    
    # Create grid for visualization
    theta = np.linspace(0, np.pi, 50)
    phi = np.linspace(0, 2*np.pi, 50)
    theta, phi = np.meshgrid(theta, phi)
    
    # Convert to Cartesian coordinates
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    # Create figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    def update(frame):
        ax.clear()
        # Reconstruct shape for this time point
        r = reconstruct_shape(coefficients[frame], theta, phi)
        
        # Plot surface
        ax.plot_surface(x*r, y*r, z*r, cmap='viridis')
        ax.set_title(f'Time point: {time_points[frame]}')
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.set_zlim([-2, 2])
    
    # Create animation
    anim = animation.FuncAnimation(fig, update, frames=len(time_points), 
                                 interval=200, blit=False)
    
    # Save animation
    anim.save(output_file, writer='ffmpeg', fps=5)
    plt.close()

if __name__ == '__main__':
    # Example usage
    lineage_dir = 'DATA/spharm/MSpppppp'  # You can change this to any lineage directory
    create_animation(lineage_dir)
