#!/usr/bin/env python3
"""
Dorsal intercalation analysis script for generating cell_plot visualizations.
This script processes dorsal intercalation cells to generate:
1. Co-clustering probability heatmap
2. Cell trajectory visualization
3. Morphological irregularity dynamics
4. Velocity field analysis
"""

from typing import Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import nibabel as nib
from scipy import ndimage
import warnings
from nibabel.nifti1 import load

warnings.filterwarnings("ignore")


class DorsalIntercalationAnalyzer:
    def __init__(self, data_dir="DATA"):
        self.data_dir = Path(data_dir)
        self.dorsal_cells = self.load_dorsal_cells()
        self.name_dict = self.load_name_dictionary()
        self.dorsal_cell_ids = self.find_dorsal_cell_ids()

    def load_dorsal_cells(self):
        """Load dorsal intercalation cell names."""
        dorsal_file = self.data_dir / "dorsal_intercalation.txt"
        cells = []
        with open(dorsal_file, "r") as f:
            for line in f:
                if line := line.strip():
                    cells.append(line)
        return cells

    def load_name_dictionary(self):
        """Load cell name dictionary."""
        dict_file = self.data_dir / "name_dictionary.csv"
        df = pd.read_csv(dict_file)
        # Create dictionary: cell_name -> cell_id
        return dict(zip(df.iloc[:, 1], df.iloc[:, 0]))

    def find_dorsal_cell_ids(self):
        """Find cell IDs for dorsal intercalation cells."""
        cell_ids = []
        print(f"Looking for {len(self.dorsal_cells)} dorsal cells")
        print(f"Name dictionary has {len(self.name_dict)} entries")

        for cell_name in self.dorsal_cells:
            if cell_name in self.name_dict:
                cell_ids.append(int(self.name_dict[cell_name]))
                print(f"Found: {cell_name} -> {int(self.name_dict[cell_name])}")
            else:
                print(f"Not found: {cell_name}")
        return cell_ids

    def load_segmentation_data(self, time_point):
        """Load segmentation data for a specific time point."""
        seg_file = (
            self.data_dir
            / f"SegmentCellUnified/WT_Sample1LabelUnified/WT_Sample1LabelUnified_{time_point:03d}_segCell.nii.gz"
        )
        return load(seg_file).get_fdata() if seg_file.exists() else None

    def extract_cell_centroids(self, time_range=(220, 251)):
        """Extract cell centroids for time range."""
        centroids = {}

        for t in range(time_range[0], time_range[1]):
            seg_data = self.load_segmentation_data(t)
            if seg_data is not None:
                time_centroids = {}
                for cell_id in self.dorsal_cell_ids:
                    mask = seg_data == cell_id
                    if np.any(mask):
                        centroid = ndimage.center_of_mass(mask)
                        time_centroids[cell_id] = centroid
                centroids[t] = time_centroids

        return centroids

    def calculate_trajectories(self):
        """Calculate cell trajectories."""
        centroids = self.extract_cell_centroids()
        trajectories = {}

        for cell_id in self.dorsal_cell_ids:
            trajectory = []
            trajectory.extend(
                [t] + list(centroids[t][cell_id])
                for t in sorted(centroids.keys())
                if cell_id in centroids[t]
            )
            if trajectory:
                trajectories[cell_id] = np.array(trajectory)

        return trajectories

    def load_clustering_results(self):
        """Load clustering results from sub_triclusters."""
        cluster_file = self.data_dir / "sub_triclusters/all_subcluster_results.json"
        if cluster_file.exists():
            with open(cluster_file, "r") as f:
                return json.load(f)
        return {}

    def calculate_coclustering_probabilities(self):
        """Calculate co-clustering probabilities for dorsal cells."""
        clustering_results = self.load_clustering_results()
        time_range = range(220, 251)

        # Create probability matrix: cells x time
        prob_matrix = np.zeros((len(self.dorsal_cell_ids), len(time_range)))

        # Calculate co-clustering probabilities based on actual clustering results
        for i, cell_id in enumerate(self.dorsal_cell_ids):
            for j, t in enumerate(time_range):
                # Calculate probability that this cell is co-clustered at time t
                prob = 0.0
                count = 0

                # Look through all clustering results for this cell and time
                for key, result in clustering_results.items():
                    if f"cell_{cell_id}" in key:
                        # Check if this time point is within the clustering result
                        time_labels = result.get("time_labels", [])
                        if len(time_labels) > (t - 220):  # Adjust for time offset
                            time_idx = t - 220
                            if time_idx >= 0 and time_idx < len(time_labels):
                                # If time_label is 1, it means this cell is part of a cluster at this time
                                prob += time_labels[time_idx]
                                count += 1

                # Average probability across all relevant clustering results
                if count > 0:
                    prob_matrix[i, j] = prob / count
                else:
                    # If no clustering data available, use low probability
                    prob_matrix[i, j] = 0.1

        return prob_matrix, time_range

    def calculate_morphological_irregularity(self, time_range=(220, 251)):
        """Calculate morphological irregularity for each cell over time."""
        irregularity_data = {}

        for cell_id in self.dorsal_cell_ids:
            irregularities = []
            times = []

            for t in range(time_range[0], time_range[1]):
                seg_data = self.load_segmentation_data(t)
                if seg_data is not None:
                    mask = seg_data == cell_id
                    if np.any(mask):
                        # Calculate irregularity as ratio of surface area to volume for 3D
                        from skimage import measure

                        if props := measure.regionprops(mask.astype(int)):
                            # For 3D data, use surface area / volume ratio
                            area = props[0].area  # This is volume for 3D
                            # Use sqrt(area) as a proxy for surface area since perimeter is not available
                            surface_proxy = np.sqrt(area)
                            if area > 0:
                                irregularity = surface_proxy / (
                                    area ** (1 / 3)
                                )  # Normalized by cube root of volume
                                irregularities.append(irregularity)
                                times.append(t)

            if irregularities:
                irregularity_data[cell_id] = {
                    "times": times,
                    "irregularities": irregularities,
                }

        return irregularity_data

    def calculate_midline_velocity(self):
        """Calculate velocity across midline."""
        trajectories = self.calculate_trajectories()
        velocities = {}

        for cell_id, trajectory in trajectories.items():
            if len(trajectory) > 1:
                # Calculate velocity in x-direction (assuming x is anterior-posterior)
                times = trajectory[:, 0]
                x_coords = trajectory[:, 1]  # Assuming x is first spatial coordinate

                # Calculate velocity
                dt = np.diff(times)
                dx = np.diff(x_coords)
                velocity = dx / dt

                velocities[cell_id] = {
                    "times": times[1:],  # Remove first time point
                    "velocity": velocity,
                }

        return velocities

    def create_demo_coclustering_data(self):
        """Create demo co-clustering data with ideal patterns."""
        # Time range: 220-255 minutes (discrete integer minutes)
        time_range = np.arange(220, 256)
        n_cells_per_side = 6  # 6 cells per side

        # Create probability matrices for left and right sides separately
        left_matrix = np.zeros((n_cells_per_side, len(time_range)))
        right_matrix = np.zeros((n_cells_per_side, len(time_range)))

        # Clustering time periods
        cluster_rise_start = 225
        cluster_rise_end = 230
        cluster_high_start = 230
        cluster_high_end = 254
        cluster_fall_start = 255

        for i in range(n_cells_per_side):
            for j, t in enumerate(time_range):
                if cluster_rise_start <= t <= cluster_rise_end:
                    # 225-230分钟快速上升
                    progress = (t - cluster_rise_start) / (
                        cluster_rise_end - cluster_rise_start
                    )
                    base_prob = 0.2 + 0.75 * progress  # 0.2到0.95
                elif cluster_high_start < t < cluster_fall_start:
                    # 230-254分钟保持高概率
                    base_prob = 0.95 + np.random.normal(0, 0.02)
                elif t == cluster_fall_start:
                    # 255分钟概率略有下降
                    base_prob = 0.8 + np.random.normal(0, 0.05)
                else:
                    # 其他时间低概率
                    base_prob = 0.1 + np.random.normal(0, 0.02)

                # 加入细胞间微小噪声
                noise = np.random.normal(0, 0.03)
                left_matrix[i, j] = base_prob + noise
                right_matrix[i, j] = base_prob + noise

        # Ensure probabilities are in [0, 1] range
        left_matrix = np.clip(left_matrix, 0, 1)
        right_matrix = np.clip(right_matrix, 0, 1)

        return left_matrix, right_matrix, time_range

    def plot_coclustering_heatmap(
        self,
        save_path: Union[str, Path] = "coclustering_heatmap.png",
        use_demo_data: bool = False,
    ):
        """Generate co-clustering probability heatmap."""
        if use_demo_data:
            # Use demo data for ideal visualization
            left_matrix, right_matrix, time_range = self.create_demo_coclustering_data()

            # Create separate left/right heatmaps
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

            # Left side heatmap
            sns.heatmap(
                left_matrix,
                xticklabels=[str(t) if t % 5 == 0 else "" for t in time_range],
                yticklabels=[f"L{i+1:02d}" for i in range(left_matrix.shape[0])],
                cmap="RdBu_r",
                vmin=0,
                vmax=1,
                cbar_kws={"label": "Co-clustering Probability", "shrink": 0.8},
                ax=ax1,
            )
            ax1.set_xlabel("Time (minutes)")
            ax1.set_ylabel("Left Dorsal Cells")

            # Right side heatmap
            sns.heatmap(
                right_matrix,
                xticklabels=[str(t) if t % 5 == 0 else "" for t in time_range],
                yticklabels=[f"R{i+1:02d}" for i in range(right_matrix.shape[0])],
                cmap="RdBu_r",
                vmin=0,
                vmax=1,
                cbar_kws={"label": "Co-clustering Probability", "shrink": 0.8},
                ax=ax2,
            )
            ax2.set_xlabel("Time (minutes)")
            ax2.set_ylabel("Right Dorsal Cells")

            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
            return save_path
        else:
            # Original implementation for real data
            if len(self.dorsal_cell_ids) == 0:
                print("No dorsal cells found, skipping heatmap")
                return None

            prob_matrix, time_range = self.calculate_coclustering_probabilities()

            plt.figure(figsize=(12, 8))

            # Create heatmap
            sns.heatmap(
                prob_matrix,
                xticklabels=[
                    str(t) for t in time_range[::5]
                ],  # Show every 5th time point
                yticklabels=[f"DC{i+1:02d}" for i in range(len(self.dorsal_cell_ids))],
                cmap="viridis",
                cbar_kws={"label": "Co-clustering Probability"},
            )

            plt.xlabel("Time (minutes post-fertilization)")
            plt.ylabel("Dorsal Cell ID")
            return self._save_velocity_field_plot(save_path)

    def plot_cell_trajectories(
        self, save_path: Union[str, Path] = "cell_trajectories.png"
    ):
        """Generate cell trajectory visualization."""
        if len(self.dorsal_cell_ids) == 0:
            print("No dorsal cells found, skipping trajectories")
            return None

        trajectories = self.calculate_trajectories()

        if not trajectories:
            print("No trajectory data found, skipping trajectories")
            return None

        plt.figure(figsize=(10, 8))

        colors = plt.cm.get_cmap("tab20")(np.linspace(0, 1, len(trajectories)))

        for i, (cell_id, trajectory) in enumerate(trajectories.items()):
            if len(trajectory) > 1:
                # Plot trajectory
                plt.plot(
                    trajectory[:, 2],
                    trajectory[:, 3],
                    color=colors[i],
                    alpha=0.7,
                    linewidth=2,
                    label=f"Cell {cell_id}",
                )

                # Mark start and end points
                plt.scatter(
                    trajectory[0, 2],
                    trajectory[0, 3],
                    color="green",
                    s=100,
                    marker="o",
                    alpha=0.8,
                    zorder=5,
                )
                plt.scatter(
                    trajectory[-1, 2],
                    trajectory[-1, 3],
                    color="red",
                    s=100,
                    marker="s",
                    alpha=0.8,
                    zorder=5,
                )

        plt.axvline(x=0, color="black", linestyle="--", alpha=0.5, label="Midline")
        self._set_plot_labels(
            "Anterior-Posterior Axis (μm)",
            "Left-Right Axis (μm)",
            "Cell Trajectory Visualization\nDorsal Intercalation (220-250 min)",
        )
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        return self._save_velocity_field_plot(save_path)

    def plot_morphological_irregularity(
        self, save_path: Union[str, Path] = "morphological_irregularity.png"
    ):
        """Generate morphological irregularity dynamics plot."""
        irregularity_data = self.calculate_morphological_irregularity()

        plt.figure(figsize=(12, 8))

        all_irregularities = []
        all_times = []

        for cell_id, data in irregularity_data.items():
            times = data["times"]
            irregularities = data["irregularities"]

            # Plot individual cell data
            plt.plot(times, irregularities, color="lightblue", alpha=0.6, linewidth=1)

            all_times.extend(times)
            all_irregularities.extend(irregularities)

        # Calculate and plot mean ± SEM
        time_bins = np.arange(220, 251)
        mean_irregularity = []
        sem_irregularity = []

        for t in time_bins:
            time_mask = np.array(all_times) == t
            if np.any(time_mask):
                values = np.array(all_irregularities)[time_mask]
                mean_irregularity.append(np.mean(values))
                sem_irregularity.append(np.std(values) / np.sqrt(len(values)))
            else:
                mean_irregularity.append(np.nan)
                sem_irregularity.append(np.nan)

        mean_irregularity = np.array(mean_irregularity)
        sem_irregularity = np.array(sem_irregularity)

        # Plot mean with error bars
        plt.plot(
            time_bins,
            mean_irregularity,
            color="darkblue",
            linewidth=3,
            label="Mean ± SEM",
        )
        plt.fill_between(
            time_bins,
            mean_irregularity - sem_irregularity,
            mean_irregularity + sem_irregularity,
            alpha=0.3,
            color="darkblue",
        )

        # Highlight co-clustering active window (example)
        plt.axvspan(
            230, 240, alpha=0.2, color="yellow", label="Co-clustering Active Window"
        )

        self._set_plot_labels(
            "Time (minutes post-fertilization)",
            "Morphological Irregularity Index",
            "Morphological Irregularity Dynamics\nDorsal Intercalation Cells",
        )
        plt.legend()
        plt.grid(True, alpha=0.3)
        return self._save_velocity_field_plot(save_path)

    def plot_velocity_field(self, save_path: Union[str, Path] = "velocity_field.png"):
        """Generate velocity field analysis plot."""
        velocities = self.calculate_midline_velocity()

        plt.figure(figsize=(12, 8))

        # Collect all velocity data
        all_velocities = []
        all_times = []

        for cell_id, data in velocities.items():
            all_times.extend(data["times"])
            all_velocities.extend(data["velocity"])

        # Create time bins
        time_bins = np.arange(220, 250, 2)  # Every 2 minutes

        # Calculate statistics for each time bin
        velocity_stats = []
        for t in time_bins:
            time_mask = (np.array(all_times) >= t) & (np.array(all_times) < t + 2)
            if np.any(time_mask):
                bin_velocities = np.array(all_velocities)[time_mask]
                velocity_stats.append(bin_velocities)
            else:
                velocity_stats.append([])

        # Create box plot
        plt.boxplot(
            [v for v in velocity_stats if len(v) > 0],
            positions=[
                t for i, t in enumerate(time_bins) if len(velocity_stats[i]) > 0
            ],
            widths=1.5,
        )

        self._set_plot_labels(
            "Time (minutes post-fertilization)",
            "Midline Crossing Velocity (μm/min)",
            "Velocity Field Analysis\nDorsal Intercalation Cells",
        )
        plt.grid(True, alpha=0.3)
        return self._save_velocity_field_plot(save_path)

    def _set_plot_labels(self, x_axis_label, y_axis_label, plot_title):
        plt.xlabel(x_axis_label)
        plt.ylabel(y_axis_label)
        plt.title(plot_title)

    def _save_velocity_field_plot(self, save_path):
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        return save_path

    def generate_all_plots(self, output_dir="dorsal_plots"):
        """Generate all required plots using a function table (dictionary of function pointers)."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        plots = {}

        if len(self.dorsal_cell_ids) == 0:
            print("No dorsal cells found, cannot generate plots")
            return plots

        # 定义函数表，key为plot类型，value为(方法, 保存路径, 参数)
        plot_table = {
            "heatmap": (
                self.plot_coclustering_heatmap,
                output_dir / "PanelA_coclustering_heatmap.png",
                {},
            ),
            "demo_heatmap": (
                self.plot_coclustering_heatmap,
                output_dir / "demo_ideal_coclustering.png",
                {"use_demo_data": True},
            ),
            # "trajectories": (self.plot_cell_trajectories, output_dir / "PanelB_cell_trajectories.png", {}),
            # "irregularity": (self.plot_morphological_irregularity, output_dir / "PanelC_morphological_irregularity.png", {}),
            # "velocity": (self.plot_velocity_field, output_dir / "PanelD_velocity_field.png", {}),
        }

        for plot_name, (plot_func, save_path, kwargs) in plot_table.items():
            print(f"Generating {plot_name} plot...")
            result = plot_func(save_path, **kwargs)
            if result:
                plots[plot_name] = result

        return plots


def main():
    """Main function to run the analysis."""
    analyzer = DorsalIntercalationAnalyzer()

    print(f"Found {len(analyzer.dorsal_cells)} dorsal intercalation cells")
    print(f"Mapped {len(analyzer.dorsal_cell_ids)} cell IDs")

    print("\nDorsal cell IDs:", analyzer.dorsal_cell_ids)

    # Generate all plots
    plots = analyzer.generate_all_plots()

    print("\nGenerated plots:")
    for plot_type, path in plots.items():
        print(f"  {plot_type}: {path}")

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
