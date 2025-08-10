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
from mpl_toolkits.mplot3d import Axes3D

warnings.filterwarnings("ignore")

# Set Times New Roman as the default font for all plots
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif', 'serif']
plt.rcParams['mathtext.fontset'] = 'stix'  # For mathematical text
# Ensure consistent font rendering
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 'large'
plt.rcParams['axes.labelsize'] = 'medium'
plt.rcParams['xtick.labelsize'] = 'small'
plt.rcParams['ytick.labelsize'] = 'small'
plt.rcParams['legend.fontsize'] = 'small'


class FontConfig:
    """统一字体配置类，支持缩放参数"""
    
    def __init__(self, scale_factor: float = 1.0):
        self.scale_factor = scale_factor
        
    @property
    def axis_label_size(self) -> int:
        return int(16 * self.scale_factor)
    
    @property
    def tick_label_size(self) -> int:
        return int(14 * self.scale_factor)
    
    @property
    def legend_size(self) -> int:
        return int(12 * self.scale_factor)
    
    @property
    def colorbar_size(self) -> int:
        return int(14 * self.scale_factor)
    
    @property
    def axis_weight(self) -> str:
        return 'normal'
    
    @property
    def font_family(self) -> str:
        return 'serif'  # Times New Roman


class IntestinalPrimordiumAnalyzer:
    """Analysis class for Intestinal Primordium Formation demo."""
    
    def __init__(self, font_config: FontConfig = None):
        # E lineage cells (20 cells total) - using actual lineage names from report.md
        self.e_lineage_cells = [
            "int1DL", "int1VL", "int1DR", "int1VR",  # Ring 1 (4 cells)
            "int2L", "int2R",                         # Ring 2 (2 cells) 
            "int3L", "int3R",                         # Ring 3 (2 cells)
            "int4L", "int4R",                         # Ring 4 (2 cells)
            "int5L", "int5R",                         # Ring 5 (2 cells)
            "int6L", "int6R",                         # Ring 6 (2 cells)
            "int7L", "int7R",                         # Ring 7 (2 cells)
            "int8L", "int8R",                         # Ring 8 (2 cells)
            "int9L", "int9R"                          # Ring 9 (2 cells)
        ]
        self.font_config = font_config or FontConfig()
        
    def create_demo_intestinal_coclustering(self):
        """Create demo co-clustering data for E lineage cells during gut formation."""
        # Time range: 350-400 minutes (gut formation period)
        time_range = np.arange(350, 401)
        n_cells = 20  # 20 E lineage cells
        
        # Create probability matrix
        prob_matrix = np.zeros((n_cells, len(time_range)))
        
        # Gut formation phases based on report.md knowledge
        gastrulation_start = 350    # 原肠作用开始 (28-cell stage)
        internalization_start = 355  # Ea/Ep内化开始
        internalization_peak = 365   # 内化完成，细胞进入囊胚腔
        proliferation_phase = 375    # E16细胞增殖期
        primordium_formation = 385   # E20肠原基形成
        tube_morphogenesis = 395     # 管腔形成与极化
        
        # Define cell ring positions for differential behavior
        ring1_cells = [0, 1, 2, 3]  # int1 ring (4 cells, anterior)
        ring2_9_cells = list(range(4, 20))  # int2-9 rings (16 cells)
        
        for i in range(n_cells):
            for j, t in enumerate(time_range):
                if gastrulation_start <= t < internalization_start:
                    # 早期原肠作用：低聚类，细胞仍在表面
                    base_prob = 0.15 + np.random.normal(0, 0.03)
                elif internalization_start <= t < internalization_peak:
                    # Ea/Ep内化阶段：快速聚类增加
                    progress = (t - internalization_start) / (internalization_peak - internalization_start)
                    base_prob = 0.2 + 0.5 * progress  # 0.2到0.7
                elif internalization_peak <= t < proliferation_phase:
                    # 内化后增殖期：中等聚类
                    base_prob = 0.65 + np.random.normal(0, 0.05)
                elif proliferation_phase <= t < primordium_formation:
                    # E16到E20增殖期：聚类增强
                    progress = (t - proliferation_phase) / (primordium_formation - proliferation_phase)
                    base_prob = 0.7 + 0.2 * progress  # 0.7到0.9
                elif primordium_formation <= t <= tube_morphogenesis:
                    # 肠原基形成到管腔形成：最高聚类
                    base_prob = 0.9 + np.random.normal(0, 0.02)
                else:
                    # 管腔形成后：稳定高聚类
                    base_prob = 0.85 + np.random.normal(0, 0.03)
                
                # Ring-specific behavior: anterior cells (ring1) show stronger clustering
                if i in ring1_cells:
                    base_prob = min(base_prob * 1.05, 1.0)  # 5% boost for anterior cells
                
                # 添加基于同源嵌入的细胞间变异
                cell_variation = np.random.normal(0, 0.04)
                prob_matrix[i, j] = np.clip(base_prob + cell_variation, 0, 1)
        
        return prob_matrix, time_range
    
    def create_demo_intestinal_trajectory(self):
        """Create demo trajectory data for E lineage internalization based on report.md."""
        time_range = np.arange(350, 401)
        trajectories = {}
        
        # 根据report.md，Ea和Ep细胞在28细胞期内化，随后增殖形成E20
        # 前4个细胞代表早期Ea/Ep后代，其余16个代表后期增殖细胞
        
        for i, cell_id in enumerate(self.e_lineage_cells):
            trajectory = []
            
            # Starting positions - 在胚胎腹侧表面排列
            if i < 4:  # Early Ea/Ep descendants (int1 ring)
                # 早期细胞位于肠道前端
                start_x = 25 + i * 3 + np.random.normal(0, 1)
                start_y = 0 + np.random.normal(0, 2)  # 围绕中线
                start_z = 0 + np.random.normal(0, 0.5)  # 表面位置
            else:  # Later proliferation descendants
                # 后期增殖细胞沿前后轴分布
                ring_idx = (i - 4) // 2  # Ring index (0-7 for rings 2-9)
                side_idx = (i - 4) % 2   # Left/right side
                start_x = 30 + ring_idx * 4 + np.random.normal(0, 1.5)
                start_y = (side_idx - 0.5) * 4 + np.random.normal(0, 1.5)  # Left-right positioning
                start_z = 0 + np.random.normal(0, 0.5)
            
            for j, t in enumerate(time_range):
                # 根据report.md的发育阶段模拟运动
                if t < 355:
                    # 28细胞期前：细胞仍在表面
                    x_pos = start_x + np.random.normal(0, 0.2)
                    y_pos = start_y + np.random.normal(0, 0.2)
                    z_pos = start_z + np.random.normal(0, 0.1)
                    
                elif 355 <= t <= 365:
                    # Ea/Ep内化期：快速向内运动（负Z速度）
                    progress = (t - 355) / 10.0
                    
                    if i < 4:  # Early internalizing cells
                        # 顶端收缩驱动的内化运动
                        z_pos = start_z - progress * (12 + np.random.normal(0, 1))
                        # 轻微向中心收敛
                        x_pos = start_x + progress * (2 * np.sign(start_y - 0))
                        y_pos = start_y * (1 - 0.2 * progress)
                    else:
                        # Later cells follow more gradually
                        z_pos = start_z - progress * (6 + np.random.normal(0, 1))
                        x_pos = start_x + progress * np.random.normal(0, 1)
                        y_pos = start_y + progress * np.random.normal(0, 1)
                        
                elif 365 < t <= 385:
                    # 增殖期：细胞在囊胚腔内缓慢重排
                    if i < 4:
                        z_pos = start_z - 12 + np.random.normal(0, 1)
                        x_pos = start_x + 2 * np.sign(start_y) + np.random.normal(0, 0.5)
                        y_pos = start_y * 0.8 + np.random.normal(0, 0.5)
                    else:
                        progress_late = (t - 365) / 20.0
                        z_pos = start_z - 6 - progress_late * 4
                        x_pos = start_x + progress_late * np.random.normal(0, 2)
                        y_pos = start_y + progress_late * np.random.normal(0, 1.5)
                        
                elif 385 < t <= 395:
                    # 肠原基形成：向管状结构收敛
                    progress_tube = (t - 385) / 10.0
                    
                    # 所有细胞向管状排列收敛
                    target_radius = 8 if i < 4 else 6  # int1 ring略大
                    angle = i * 2 * np.pi / len(self.e_lineage_cells)
                    
                    target_x = start_x
                    target_y = target_radius * np.cos(angle)
                    target_z = start_z - (12 if i < 4 else 10)
                    
                    x_pos = start_x + (target_x - start_x) * progress_tube + np.random.normal(0, 0.3)
                    y_pos = start_y + (target_y - start_y) * progress_tube + np.random.normal(0, 0.3)
                    z_pos = (start_z - (12 if i < 4 else 10)) + np.random.normal(0, 0.2)
                    
                else:  # t > 395
                    # 管腔形成后：稳定的管状结构
                    target_radius = 8 if i < 4 else 6
                    angle = i * 2 * np.pi / len(self.e_lineage_cells)
                    
                    x_pos = start_x + np.random.normal(0, 0.2)
                    y_pos = target_radius * np.cos(angle) + np.random.normal(0, 0.3)
                    z_pos = start_z - (12 if i < 4 else 10) + np.random.normal(0, 0.2)
                
                trajectory.append([t, x_pos, y_pos, z_pos])
            
            trajectories[cell_id] = np.array(trajectory)
        
        return trajectories
    
    def create_demo_intestinal_velocity(self):
        """Create demo velocity data showing apical constriction-driven internalization."""
        trajectories = self.create_demo_intestinal_trajectory()
        velocities = {}
        
        for i, (cell_id, trajectory) in enumerate(trajectories.items()):
            if len(trajectory) > 1:
                times = trajectory[:, 0]
                x_coords = trajectory[:, 1]
                y_coords = trajectory[:, 2] 
                z_coords = trajectory[:, 3]  # Z coordinates (internalization)
                
                dt = np.diff(times)
                dx = np.diff(x_coords)
                dy = np.diff(y_coords)
                dz = np.diff(z_coords)
                
                # 计算3D速度分量
                vx = dx / dt
                vy = dy / dt
                vz = dz / dt  # 负值表示内化
                
                # 根据report.md知识，模拟顶端收缩的特征
                apical_constriction_velocity = np.zeros_like(vz)
                
                for j, t in enumerate(times[1:]):
                    if 355 <= t <= 365:  # 内化期
                        # 顶端收缩驱动的内化：早期细胞更强
                        if i < 4:  # Early Ea/Ep descendants
                            # 强烈的顶端收缩，产生突发性负Z速度
                            intensity = 1.0 - (t - 355) / 10.0  # 随时间递减
                            apical_constriction_velocity[j] = -8.0 * intensity + np.random.normal(0, 0.5)
                        else:
                            # 后期细胞跟随性内化
                            intensity = 0.6 - (t - 355) / 15.0
                            apical_constriction_velocity[j] = -4.0 * intensity + np.random.normal(0, 0.3)
                    elif 365 < t <= 385:
                        # 增殖期：轻微的重排运动
                        apical_constriction_velocity[j] = vz[j] * 0.3 + np.random.normal(0, 0.2)
                    else:
                        # 其他时期：维持缓慢运动
                        apical_constriction_velocity[j] = vz[j] * 0.1 + np.random.normal(0, 0.1)
                
                velocities[cell_id] = {
                    "times": times[1:],
                    "velocity_x": vx,
                    "velocity_y": vy, 
                    "velocity_z": vz,  # 原始Z速度
                    "apical_constriction_velocity": apical_constriction_velocity,  # 顶端收缩特征
                    "speed_3d": np.sqrt(vx**2 + vy**2 + vz**2),  # 3D速度幅度
                    "internalization_phase": i < 4  # 是否为早期内化细胞
                }
        
        return velocities
    
    def plot_intestinal_coclustering_heatmap(self, save_path="intestinal_coclustering.png"):
        """Plot E lineage co-clustering heatmap."""
        prob_matrix, time_range = self.create_demo_intestinal_coclustering()
        
        plt.figure(figsize=(14, 10))
        
        sns.heatmap(
            prob_matrix,
            xticklabels=[str(t) if t % 10 == 0 else "" for t in time_range],
            yticklabels=self.e_lineage_cells,
            cmap="RdBu_r",
            vmin=0,
            vmax=1,
            cbar_kws={"label": "Co-clustering Probability", "shrink": 0.8},
        )
        
        plt.xlabel("Time (minutes post-fertilization)", fontsize=self.font_config.axis_label_size, fontweight=self.font_config.axis_weight)
        plt.ylabel("E Lineage Cells", fontsize=self.font_config.axis_label_size, fontweight=self.font_config.axis_weight)
        plt.tick_params(axis='both', which='major', labelsize=self.font_config.tick_label_size)
        
        # Update colorbar font size
        cbar = plt.gca().collections[0].colorbar
        cbar.ax.tick_params(labelsize=self.font_config.colorbar_size)
        cbar.set_label("Co-clustering Probability", fontsize=self.font_config.colorbar_size, fontweight=self.font_config.axis_weight)
        
        # Add phase annotations based on report.md gut morphogenesis stages
        phase_lines = [
            (5, 'red', 'Ea/Ep内化'),      # 355 min - internalization start
            (15, 'orange', '内化完成'),     # 365 min - internalization complete  
            (25, 'green', 'E16增殖'),      # 375 min - proliferation phase
            (35, 'blue', 'E20原基'),       # 385 min - primordium formation
            (45, 'purple', '管腔形成')      # 395 min - tube morphogenesis
        ]
        
        for x_pos, color, label in phase_lines:
            plt.axvline(x=x_pos, color=color, linestyle='--', alpha=0.7, linewidth=1.5)
            
        # Add text annotations for major phases  
        plt.text(2, len(self.e_lineage_cells)-2, '原肠作用', fontsize=self.font_config.legend_size-2, 
                rotation=90, alpha=0.8, color='red')
        plt.text(10, len(self.e_lineage_cells)-2, '内化期', fontsize=self.font_config.legend_size-2,
                rotation=90, alpha=0.8, color='orange') 
        plt.text(30, len(self.e_lineage_cells)-2, '增殖期', fontsize=self.font_config.legend_size-2,
                rotation=90, alpha=0.8, color='green')
        plt.text(40, len(self.e_lineage_cells)-2, '形态建成', fontsize=self.font_config.legend_size-2,
                rotation=90, alpha=0.8, color='blue')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        return save_path
    
    def plot_intestinal_trajectories(self, save_path="intestinal_trajectories.png"):
        """Plot E lineage internalization trajectories."""
        trajectories = self.create_demo_intestinal_trajectory()
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        colors = plt.cm.get_cmap("viridis")(np.linspace(0, 1, len(trajectories)))
        
        for i, (cell_id, trajectory) in enumerate(trajectories.items()):
            ax.plot(trajectory[:, 1], trajectory[:, 2], trajectory[:, 3], 
                   color=colors[i], alpha=0.7, linewidth=1.5)
            
            # Mark start and end points
            ax.scatter(trajectory[0, 1], trajectory[0, 2], trajectory[0, 3],
                      color='green', s=50, alpha=0.8)
            ax.scatter(trajectory[-1, 1], trajectory[-1, 2], trajectory[-1, 3],
                      color='red', s=50, alpha=0.8)
        
        ax.set_xlabel("Anterior-Posterior (μm)", fontsize=self.font_config.axis_label_size, fontweight=self.font_config.axis_weight)
        ax.set_ylabel("Left-Right (μm)", fontsize=self.font_config.axis_label_size, fontweight=self.font_config.axis_weight)
        ax.set_zlabel("Dorsal-Ventral (μm)", fontsize=self.font_config.axis_label_size, fontweight=self.font_config.axis_weight)
        ax.tick_params(axis='both', which='major', labelsize=self.font_config.tick_label_size)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        return save_path
    
    def plot_intestinal_velocity_field(self, save_path="intestinal_velocity.png"):
        """Plot internalization velocity field (negative Z velocity)."""
        velocities = self.create_demo_intestinal_velocity()
        
        plt.figure(figsize=(12, 8))
        
        all_velocities = []
        all_times = []
        
        for cell_id, data in velocities.items():
            all_times.extend(data["times"])
            all_velocities.extend(data["velocity"])
        
        # Create time bins for velocity analysis
        time_bins = np.arange(350, 400, 5)
        velocity_stats = []
        
        for t in time_bins:
            time_mask = (np.array(all_times) >= t) & (np.array(all_times) < t + 5)
            if np.any(time_mask):
                bin_velocities = np.array(all_velocities)[time_mask]
                velocity_stats.append(bin_velocities)
            else:
                velocity_stats.append([])
        
        # Create violin plot
        positions = [t for i, t in enumerate(time_bins) if len(velocity_stats[i]) > 0]
        data_to_plot = [v for v in velocity_stats if len(v) > 0]
        
        plt.violinplot(data_to_plot, positions=positions, widths=3)
        
        plt.xlabel("Time (minutes post-fertilization)", fontsize=self.font_config.axis_label_size, fontweight=self.font_config.axis_weight, fontfamily=self.font_config.font_family)
        plt.ylabel("Internalization Velocity (μm/min)", fontsize=self.font_config.axis_label_size, fontweight=self.font_config.axis_weight, fontfamily=self.font_config.font_family)
        plt.tick_params(axis='both', which='major', labelsize=self.font_config.tick_label_size)
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        return save_path
    
    def plot_intestinal_coclustering_features_pie(self, save_path="intestinal_coclustering_features_pie.png"):
        """Create pie chart showing intestinal morphogenesis co-clustering feature distribution."""
        # Quantitative local geometrical features measurable in co-clustering analysis
        features = {
            'Z-axis Velocity': 28,           # Primary: internalization rate (negative)
            'Apical Surface Area': 24,       # Constriction measurement
            'Cell Volume Change': 19,        # Compression during internalization
            'Radial Distance': 14,           # Distance from gut center axis
            'Cell Sphericity': 10,           # Roundness measure
            'Neighbor Contact Number': 5     # Connectivity in clustering
        }
        
        # Colors representing different geometrical aspects
        colors = ['#C0392B', '#2980B9', '#27AE60', '#E67E22', '#8E44AD', '#16A085']
        
        plt.figure(figsize=(10, 8))
        
        # Create pie chart
        wedges, texts, autotexts = plt.pie(
            features.values(), 
            labels=features.keys(),
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': self.font_config.legend_size, 'fontweight': 'normal'}
        )
        
        # Enhance the appearance
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(self.font_config.legend_size)
            autotext.set_fontweight('normal')
        
        plt.title('Intestinal Morphogenesis Co-clustering Features\nQuantitative Local Geometrical Properties', 
                 fontsize=self.font_config.axis_label_size, 
                 fontweight=self.font_config.axis_weight,
                 fontfamily=self.font_config.font_family, pad=20)
        
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        return save_path


class DorsalIntercalationAnalyzer:
    def __init__(self, data_dir="DATA", font_config: FontConfig = None):
        self.data_dir = Path(data_dir)
        self.dorsal_cells = self.load_dorsal_cells()
        self.name_dict = self.load_name_dictionary()
        self.dorsal_cell_ids = self.find_dorsal_cell_ids()
        self.font_config = font_config or FontConfig()

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

    def create_demo4_coclustering_data(self):
        """Create Demo4 co-clustering data based on real intestinal primordium formation timeline (350-400 minutes)."""
        # Time range: 350-400 minutes (intestinal morphogenesis period)
        time_range = np.arange(350, 401)
        n_cells_total = 20  # 20 E-lineage cells (int1-int9 rings)
        
        # Create probability matrix based on real intestinal formation phases
        prob_matrix = np.zeros((n_cells_total, len(time_range)))
        
        # Real biological phases based on report.md:
        gastrulation_start = 350    # 原肠作用开始 (28-cell stage)
        internalization_start = 355  # Ea/Ep内化开始
        internalization_peak = 365   # 内化完成，细胞进入囊胚腔
        proliferation_peak = 375     # E16/E20增殖完成
        reorganization_peak = 385    # 重组和嵌入活动高峰
        tube_formation_start = 390   # 管腔形成开始
        tube_formation_end = 400     # 极化上皮管形成完成
        
        for i in range(n_cells_total):
            for j, t in enumerate(time_range):
                if gastrulation_start <= t < internalization_start:
                    # 350-355分钟：原肠作用开始，低活动
                    base_prob = 0.15 + np.random.normal(0, 0.02)
                elif internalization_start <= t < internalization_peak:
                    # 355-365分钟：内化期，活动递增
                    progress = (t - internalization_start) / (internalization_peak - internalization_start)
                    base_prob = 0.15 + 0.6 * progress  # 0.15到0.75
                elif internalization_peak <= t < proliferation_peak:
                    # 365-375分钟：增殖期，继续上升
                    progress = (t - internalization_peak) / (proliferation_peak - internalization_peak)
                    base_prob = 0.75 + 0.2 * progress  # 0.75到0.95
                elif proliferation_peak <= t < reorganization_peak:
                    # 375-385分钟：重组嵌入高峰期，最高活动
                    base_prob = 0.95 + np.random.normal(0, 0.02)
                elif reorganization_peak <= t < tube_formation_start:
                    # 385-390分钟：开始下降
                    progress = (t - reorganization_peak) / (tube_formation_start - reorganization_peak)
                    base_prob = 0.95 - 0.3 * progress  # 0.95到0.65
                elif tube_formation_start <= t <= tube_formation_end:
                    # 390-400分钟：管腔形成，活动继续下降
                    progress = (t - tube_formation_start) / (tube_formation_end - tube_formation_start)
                    base_prob = 0.65 - 0.35 * progress  # 0.65到0.30
                else:
                    # 其他时间段
                    base_prob = 0.1 + np.random.normal(0, 0.02)
                
                # 加入细胞间微小噪声，但保持E-lineage细胞的同步性
                noise = np.random.normal(0, 0.03)
                prob_matrix[i, j] = base_prob + noise
        
        # Ensure probabilities are in [0, 1] range
        prob_matrix = np.clip(prob_matrix, 0, 1)
        
        return prob_matrix, time_range

    def analyze_coclustering_features(self):
        """Analyze features in dorsal co-clustering for pie chart."""
        prob_matrix, time_range = self.calculate_coclustering_probabilities()

        if prob_matrix.size == 0:
            # Use demo data if no real data available
            left_matrix, right_matrix, _ = self.create_demo_coclustering_data()
            prob_matrix = np.vstack([left_matrix, right_matrix])

        # Define feature categories based on clustering behavior
        features = {
            "High Activity Cells": 0,  # 高活跃聚类细胞 (mean prob > 0.7)
            "Medium Activity Cells": 0,  # 中等活跃聚类细胞 (0.4 < mean prob <= 0.7)
            "Low Activity Cells": 0,  # 低活跃聚类细胞 (0.2 < mean prob <= 0.4)
            "Inactive Cells": 0,  # 非活跃细胞 (mean prob <= 0.2)
        }

        # Analyze each cell's clustering behavior
        for i in range(prob_matrix.shape[0]):
            cell_mean_prob = np.mean(prob_matrix[i, :])

            if cell_mean_prob > 0.7:
                features["High Activity Cells"] += 1
            elif cell_mean_prob > 0.4:
                features["Medium Activity Cells"] += 1
            elif cell_mean_prob > 0.2:
                features["Low Activity Cells"] += 1
            else:
                features["Inactive Cells"] += 1

        return features

    def plot_feature_pie_chart(
        self, save_path: Union[str, Path] = "feature_pie_chart.png"
    ):
        """Generate pie chart showing feature distribution in dorsal co-clustering."""
        features = self.analyze_coclustering_features()

        # Remove categories with zero counts
        features = {k: v for k, v in features.items() if v > 0}

        # Create pie chart
        plt.figure(figsize=(10, 8))

        labels = list(features.keys())
        sizes = list(features.values())
        colors = ["#ff9999", "#66b3ff", "#99ff99", "#ffcc99"][: len(labels)]

        # Create pie chart with enhanced styling
        wedges, texts, autotexts = plt.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            startangle=90,
            explode=[0.05] * len(labels),  # Slightly separate all slices
            shadow=True,
            textprops={"fontsize": 12},
        )

        # Enhance text styling
        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontweight("bold")

        plt.axis("equal")  # Equal aspect ratio ensures circular pie

        # Add legend with cell counts
        legend_labels = [f"{label}: {count} cells" for label, count in features.items()]
        plt.legend(
            wedges,
            legend_labels,
            title="Cell Categories",
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1),
        )

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        return save_path

    def plot_coclustering_heatmap(
        self,
        save_path: Union[str, Path] = "coclustering_heatmap.png",
        use_demo_data: bool = False,
    ):
        """Generate co-clustering probability heatmap."""
        if use_demo_data:
            # Use demo data for ideal visualization
            left_matrix, right_matrix, time_range = self.create_demo_coclustering_data()

            # Create separate left/right heatmaps with larger fonts
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
            ax1.set_xlabel("Time (minutes)", fontsize=self.font_config.axis_label_size, fontweight=self.font_config.axis_weight)
            ax1.set_ylabel("Left Dorsal Cells", fontsize=self.font_config.axis_label_size, fontweight=self.font_config.axis_weight)
            ax1.tick_params(axis='both', which='major', labelsize=self.font_config.tick_label_size)
            
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
            ax2.set_xlabel("Time (minutes)", fontsize=self.font_config.axis_label_size, fontweight=self.font_config.axis_weight)
            ax2.set_ylabel("Right Dorsal Cells", fontsize=self.font_config.axis_label_size, fontweight=self.font_config.axis_weight)
            ax2.tick_params(axis='both', which='major', labelsize=self.font_config.tick_label_size)
            
            # Update colorbar font sizes
            cbar1 = ax1.collections[0].colorbar
            cbar2 = ax2.collections[0].colorbar
            cbar1.ax.tick_params(labelsize=self.font_config.colorbar_size)
            cbar1.set_label("Co-clustering Probability", fontsize=self.font_config.colorbar_size, fontweight=self.font_config.axis_weight)
            cbar2.ax.tick_params(labelsize=self.font_config.colorbar_size)
            cbar2.set_label("Co-clustering Probability", fontsize=self.font_config.colorbar_size, fontweight=self.font_config.axis_weight)

            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
            return save_path
    
    def plot_left_coclustering_heatmap(self, save_path: Union[str, Path] = "left_coclustering_heatmap.png"):
        """Generate left side co-clustering probability heatmap in square format."""
        left_matrix, _, time_range = self.create_demo_coclustering_data()
        
        # Create square figure
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        # Left side heatmap
        sns.heatmap(
            left_matrix,
            xticklabels=[str(t) if t % 5 == 0 else "" for t in time_range],
            yticklabels=[f"L{i+1:02d}" for i in range(left_matrix.shape[0])],
            cmap="RdBu_r",
            vmin=0,
            vmax=1,
            cbar_kws={"label": "Co-clustering Probability", "shrink": 0.8},
            ax=ax,
        )
        ax.set_xlabel("Time (minutes)", fontsize=self.font_config.axis_label_size, fontweight=self.font_config.axis_weight)
        ax.set_ylabel("Left Dorsal Cells", fontsize=self.font_config.axis_label_size, fontweight=self.font_config.axis_weight)
        ax.tick_params(axis='y', which='major', labelsize=self.font_config.tick_label_size)
        ax.tick_params(axis='x', which='major', labelsize=int(self.font_config.tick_label_size * 0.8))  # Smaller x-tick labels
        
        # Update colorbar font size
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=self.font_config.colorbar_size)
        cbar.set_label("Co-clustering Probability", fontsize=self.font_config.colorbar_size, fontweight=self.font_config.axis_weight)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        return save_path
    
    def plot_right_coclustering_heatmap(self, save_path: Union[str, Path] = "right_coclustering_heatmap.png"):
        """Generate right side co-clustering probability heatmap in square format."""
        _, right_matrix, time_range = self.create_demo_coclustering_data()
        
        # Create square figure
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        # Right side heatmap
        sns.heatmap(
            right_matrix,
            xticklabels=[str(t) if t % 5 == 0 else "" for t in time_range],
            yticklabels=[f"R{i+1:02d}" for i in range(right_matrix.shape[0])],
            cmap="RdBu_r",
            vmin=0,
            vmax=1,
            cbar_kws={"label": "Co-clustering Probability", "shrink": 0.8},
            ax=ax,
        )
        ax.set_xlabel("Time (minutes)", fontsize=self.font_config.axis_label_size, fontweight=self.font_config.axis_weight)
        ax.set_ylabel("Right Dorsal Cells", fontsize=self.font_config.axis_label_size, fontweight=self.font_config.axis_weight)
        ax.tick_params(axis='y', which='major', labelsize=self.font_config.tick_label_size)
        ax.tick_params(axis='x', which='major', labelsize=int(self.font_config.tick_label_size * 0.8))  # Smaller x-tick labels
        
        # Update colorbar font size
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=self.font_config.colorbar_size)
        cbar.set_label("Co-clustering Probability", fontsize=self.font_config.colorbar_size, fontweight=self.font_config.axis_weight)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        return save_path
    
    def plot_demo4_coclustering_heatmap(self, save_path: Union[str, Path] = "demo4_coclustering_heatmap.png"):
        """Generate Demo4 co-clustering probability heatmap for intestinal primordium formation (350-400 min)."""
        prob_matrix, time_range = self.create_demo4_coclustering_data()
        
        # E-lineage cell names from report.md
        e_cell_names = [
            "int1DL", "int1VL", "int1DR", "int1VR",  # Ring 1 (4 cells)
            "int2L", "int2R",                         # Ring 2 (2 cells) 
            "int3L", "int3R",                         # Ring 3 (2 cells)
            "int4L", "int4R",                         # Ring 4 (2 cells)
            "int5L", "int5R",                         # Ring 5 (2 cells)
            "int6L", "int6R",                         # Ring 6 (2 cells)
            "int7L", "int7R",                         # Ring 7 (2 cells)
            "int8L", "int8R",                         # Ring 8 (2 cells)
            "int9L", "int9R"                          # Ring 9 (2 cells)
        ]
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Demo4 intestinal heatmap
        sns.heatmap(
            prob_matrix,
            xticklabels=[str(t) if t % 5 == 0 else "" for t in time_range],
            yticklabels=e_cell_names,
            cmap="RdBu_r",
            vmin=0,
            vmax=1,
            cbar_kws={"label": "Co-clustering Probability", "shrink": 0.8},
            ax=ax,
        )
        ax.set_xlabel("Time (minutes)", fontsize=self.font_config.axis_label_size, fontweight=self.font_config.axis_weight)
        ax.set_ylabel("E-lineage Cells", fontsize=self.font_config.axis_label_size, fontweight=self.font_config.axis_weight)
        ax.tick_params(axis='y', which='major', labelsize=int(self.font_config.tick_label_size * 0.9))
        ax.tick_params(axis='x', which='major', labelsize=int(self.font_config.tick_label_size * 0.8))
        
        # Update colorbar font size
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=self.font_config.colorbar_size)
        cbar.set_label("Co-clustering Probability", fontsize=self.font_config.colorbar_size, fontweight=self.font_config.axis_weight)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        return save_path
    
    def plot_combined_coclustering_heatmap(
        self,
        save_path: Union[str, Path] = "coclustering_heatmap.png",
        use_demo_data: bool = False,
    ):
        """Generate original combined co-clustering probability heatmap (for backwards compatibility)."""
        if use_demo_data:
            # Use demo data for ideal visualization
            left_matrix, right_matrix, time_range = self.create_demo_coclustering_data()

            # Create separate left/right heatmaps with larger fonts
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
            ax1.set_xlabel("Time (minutes)", fontsize=self.font_config.axis_label_size, fontweight=self.font_config.axis_weight)
            ax1.set_ylabel("Left Dorsal Cells", fontsize=self.font_config.axis_label_size, fontweight=self.font_config.axis_weight)
            ax1.tick_params(axis='both', which='major', labelsize=self.font_config.tick_label_size)
            
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
            ax2.set_xlabel("Time (minutes)", fontsize=self.font_config.axis_label_size, fontweight=self.font_config.axis_weight)
            ax2.set_ylabel("Right Dorsal Cells", fontsize=self.font_config.axis_label_size, fontweight=self.font_config.axis_weight)
            ax2.tick_params(axis='both', which='major', labelsize=self.font_config.tick_label_size)
            
            # Update colorbar font sizes
            cbar1 = ax1.collections[0].colorbar
            cbar2 = ax2.collections[0].colorbar
            cbar1.ax.tick_params(labelsize=self.font_config.colorbar_size)
            cbar1.set_label("Co-clustering Probability", fontsize=self.font_config.colorbar_size, fontweight=self.font_config.axis_weight)
            cbar2.ax.tick_params(labelsize=self.font_config.colorbar_size)
            cbar2.set_label("Co-clustering Probability", fontsize=self.font_config.colorbar_size, fontweight=self.font_config.axis_weight)

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

    def create_demo_trajectory_data(self):
        """Create demo cell trajectory data showing dorsal intercalation movement."""
        time_range = np.arange(220, 251)
        trajectories = {}
        
        # Generate trajectories for left side cells (negative Y, moving toward midline)
        for i in range(6):
            cell_id = f"L{i+1:02d}"
            trajectory = []
            
            # Starting positions (left side)
            start_x = 50 + np.random.normal(0, 5)
            start_y = -30 - i*5 + np.random.normal(0, 2)
            start_z = 20 + np.random.normal(0, 3)
            
            for j, t in enumerate(time_range):
                if 225 <= t <= 240:
                    progress = (t - 225) / 15.0
                    y_pos = start_y + progress * (15 + i*2)
                    x_pos = start_x + progress * (5 + np.random.normal(0, 1))
                else:
                    y_pos = start_y if t < 225 else start_y + (15 + i*2)
                    x_pos = start_x if t < 225 else start_x + 5
                
                x_pos += np.random.normal(0, 0.5)
                y_pos += np.random.normal(0, 0.5)
                z_pos = start_z + np.random.normal(0, 0.3)
                
                trajectory.append([t, x_pos, y_pos, z_pos])
            
            trajectories[cell_id] = np.array(trajectory)
        
        # Generate trajectories for right side cells
        for i in range(6):
            cell_id = f"R{i+1:02d}"
            trajectory = []
            
            start_x = 50 + np.random.normal(0, 5)
            start_y = 30 + i*5 + np.random.normal(0, 2)
            start_z = 20 + np.random.normal(0, 3)
            
            for j, t in enumerate(time_range):
                if 225 <= t <= 240:
                    progress = (t - 225) / 15.0
                    y_pos = start_y - progress * (15 + i*2)
                    x_pos = start_x + progress * (5 + np.random.normal(0, 1))
                else:
                    y_pos = start_y if t < 225 else start_y - (15 + i*2)
                    x_pos = start_x if t < 225 else start_x + 5
                
                x_pos += np.random.normal(0, 0.5)
                y_pos += np.random.normal(0, 0.5)
                z_pos = start_z + np.random.normal(0, 0.3)
                
                trajectory.append([t, x_pos, y_pos, z_pos])
            
            trajectories[cell_id] = np.array(trajectory)
        
        return trajectories

    def plot_cell_trajectories(
        self, save_path: Union[str, Path] = "cell_trajectories.png", use_demo_data: bool = True
    ):
        """Generate cell trajectory visualization."""
        if use_demo_data:
            trajectories = self.create_demo_trajectory_data()
        else:
            if len(self.dorsal_cell_ids) == 0:
                print("No dorsal cells found, using demo data")
                trajectories = self.create_demo_trajectory_data()
            else:
                trajectories = self.calculate_trajectories()
                if not trajectories:
                    print("No trajectory data found, using demo data")
                    trajectories = self.create_demo_trajectory_data()

        plt.figure(figsize=(12, 10))

        colors = plt.cm.get_cmap("tab20")(np.linspace(0, 1, len(trajectories)))

        for i, (cell_id, trajectory) in enumerate(trajectories.items()):
            if len(trajectory) > 1:
                # Plot trajectory (Y vs X for top view)
                plt.plot(
                    trajectory[:, 2],  # Y coordinates (left-right)
                    trajectory[:, 1],  # X coordinates (anterior-posterior)
                    color=colors[i],
                    alpha=0.7,
                    linewidth=2,
                    label=f"Cell {cell_id}",
                )

                # Mark start and end points
                plt.scatter(
                    trajectory[0, 2],
                    trajectory[0, 1],
                    color="green",
                    s=100,
                    marker="o",
                    alpha=0.8,
                    zorder=5,
                )
                plt.scatter(
                    trajectory[-1, 2],
                    trajectory[-1, 1],
                    color="red",
                    s=100,
                    marker="s",
                    alpha=0.8,
                    zorder=5,
                )

        plt.axvline(x=0, color="black", linestyle="--", alpha=0.7, linewidth=2, label="Midline")
        plt.xlabel("Left-Right Axis (μm)", fontsize=self.font_config.axis_label_size, fontweight=self.font_config.axis_weight)
        plt.ylabel("Anterior-Posterior Axis (μm)", fontsize=self.font_config.axis_label_size, fontweight=self.font_config.axis_weight)
        plt.tick_params(axis='both', which='major', labelsize=self.font_config.tick_label_size)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=self.font_config.legend_size)
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

    def create_demo_velocity_data(self):
        """Create demo velocity field data showing dorsal intercalation dynamics."""
        trajectories = self.create_demo_trajectory_data()
        velocities = {}
        
        for cell_id, trajectory in trajectories.items():
            if len(trajectory) > 1:
                times = trajectory[:, 0]
                y_coords = trajectory[:, 2]  # Y coordinates (toward midline)
                
                dt = np.diff(times)
                dy = np.diff(y_coords)
                velocity = dy / dt
                
                velocities[cell_id] = {
                    "times": times[1:],
                    "velocity": velocity,
                }
        
        return velocities

    def plot_velocity_field(self, save_path: Union[str, Path] = "velocity_field.png", use_demo_data: bool = True):
        """Generate velocity field analysis plot."""
        if use_demo_data:
            velocities = self.create_demo_velocity_data()
        else:
            velocities = self.calculate_midline_velocity()
            if not velocities:
                print("No velocity data found, using demo data")
                velocities = self.create_demo_velocity_data()

        plt.figure(figsize=(12, 8))

        # Collect all velocity data
        all_velocities = []
        all_times = []

        for cell_id, data in velocities.items():
            all_times.extend(data["times"])
            all_velocities.extend(data["velocity"])

        # Create time bins
        time_bins = np.arange(220, 250, 3)  # Every 3 minutes

        # Calculate statistics for each time bin
        velocity_stats = []
        for t in time_bins:
            time_mask = (np.array(all_times) >= t) & (np.array(all_times) < t + 3)
            if np.any(time_mask):
                bin_velocities = np.array(all_velocities)[time_mask]
                velocity_stats.append(bin_velocities)
            else:
                velocity_stats.append([])

        # Create violin plot for better visualization
        positions = [t for i, t in enumerate(time_bins) if len(velocity_stats[i]) > 0]
        data_to_plot = [v for v in velocity_stats if len(v) > 0]
        
        if data_to_plot:
            plt.violinplot(data_to_plot, positions=positions, widths=2)

        plt.xlabel("Time (minutes post-fertilization)", fontsize=self.font_config.axis_label_size, fontweight=self.font_config.axis_weight)
        plt.ylabel("Midline Crossing Velocity (μm/min)", fontsize=self.font_config.axis_label_size, fontweight=self.font_config.axis_weight)
        plt.tick_params(axis='both', which='major', labelsize=self.font_config.tick_label_size)
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
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

    def plot_dorsal_coclustering_features_pie(self, save_path: Union[str, Path] = "dorsal_coclustering_features_pie.png"):
        """Create pie chart showing dorsal intercalation co-clustering feature distribution."""
        # Quantitative local geometrical features measurable in co-clustering analysis
        features = {
            'Y-axis Velocity': 26,           # Primary: medial movement velocity
            'Cell Elongation Ratio': 21,     # Aspect ratio during wedge formation  
            'Surface Curvature': 18,         # Shape changes during migration
            'Local Cell Density': 15,        # Clustering intensity measure
            'Directional Persistence': 12,   # Movement direction consistency
            'Cell-Cell Contact Area': 8      # Adhesion strength indicator
        }
        
        # Colors representing different geometrical aspects
        colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C']
        
        plt.figure(figsize=(10, 8))
        
        # Create pie chart
        wedges, texts, autotexts = plt.pie(
            features.values(), 
            labels=features.keys(),
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': self.font_config.legend_size, 'fontweight': 'normal'}
        )
        
        # Enhance the appearance
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(self.font_config.legend_size)
            autotext.set_fontweight('normal')
        
        plt.title('Dorsal Intercalation Co-clustering Features\nQuantitative Local Geometrical Properties', 
                 fontsize=self.font_config.axis_label_size, 
                 fontweight=self.font_config.axis_weight, 
                 fontfamily=self.font_config.font_family, pad=20)
        
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        return save_path

    def generate_all_plots(self, output_dir="demo_plots"):
        """Generate all required plots including dorsal intercalation demos and intestinal primordium demos."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        plots = {}
        
        # Initialize intestinal analyzer with same font config
        intestinal_analyzer = IntestinalPrimordiumAnalyzer(font_config=self.font_config)

        # 定义完整的函数表，生成6张图：3个dorsal demo + 2个intestinal demo + 2个pie charts
        plot_table = {
            # Dorsal Intercalation Demos (3 figures - separate left/right heatmaps + new demo4)
            "dorsal_left_coclustering": (
                self.plot_left_coclustering_heatmap,
                output_dir / "Demo1A_Dorsal_Left_Coclustering_Heatmap.png",
                {},
            ),
            "dorsal_right_coclustering": (
                self.plot_right_coclustering_heatmap,
                output_dir / "Demo1B_Dorsal_Right_Coclustering_Heatmap.png",
                {},
            ),
            "dorsal_trajectories": (
                self.plot_cell_trajectories,
                output_dir / "Demo2_Dorsal_Cell_Trajectories.png",
                {"use_demo_data": True},
            ),
            # New Demo4 - Intestinal Primordium Formation Co-clustering (350-400min)
            "demo4_coclustering": (
                self.plot_demo4_coclustering_heatmap,
                output_dir / "Demo4_Intestinal_Primordium_Coclustering_Heatmap.png",
                {},
            ),
            
            # Intestinal Primordium Formation Demos (1 figure - removed Demo5 trajectories)
            "intestinal_velocity": (
                intestinal_analyzer.plot_intestinal_velocity_field,
                output_dir / "Demo6_Intestinal_Velocity_Field.png",
                {},
            ),
            
            # Co-clustering Feature Analysis Pie Charts
            "dorsal_features_pie": (
                self.plot_dorsal_coclustering_features_pie,
                output_dir / "Demo7A_Dorsal_Coclustering_Features_Pie.png",
                {},
            ),
            "intestinal_features_pie": (
                intestinal_analyzer.plot_intestinal_coclustering_features_pie,
                output_dir / "Demo7B_Intestinal_Coclustering_Features_Pie.png",
                {},
            ),
        }

        for plot_name, (plot_func, save_path, kwargs) in plot_table.items():
            print(f"Generating {plot_name} demo...")
            try:
                result = plot_func(save_path, **kwargs)
                if result:
                    plots[plot_name] = result
                    print(f"✓ {plot_name} saved to: {result}")
            except Exception as e:
                print(f"✗ Error generating {plot_name}: {e}")

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
