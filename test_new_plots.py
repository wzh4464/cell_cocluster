#!/usr/bin/env python3
"""
Test script for the three new plot functions following detailed requirements.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# 设置字体
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif', 'serif']
plt.rcParams['mathtext.fontset'] = 'stix'

class TestPlotGenerator:
    def __init__(self):
        self.output_dir = Path("test_new_plots")
        self.output_dir.mkdir(exist_ok=True)
        
        # 字体配置
        self.axis_label_size = 14
        self.tick_label_size = 12  
        self.legend_size = 10
        self.colorbar_size = 12
        
    def create_fig1_dorsal_consensus(self):
        """Fig.1 背侧一致性共关联矩阵 (12×12)"""
        
        # 构建模拟真实的共关联矩阵
        np.random.seed(42)
        cell_labels = [f"L{i+1:02d}" for i in range(6)] + [f"R{i+1:02d}" for i in range(6)]
        
        # 定义3个生物学簇
        clusters = {
            1: [0, 1, 6, 7],           # L01,L02,R01,R02 (前端)
            2: [2, 3, 4, 8, 9, 10],    # L03-L05,R03-R05 (中段) 
            3: [5, 11]                 # L06,R06 (后端)
        }
        
        # 构建共关联矩阵
        C = np.eye(12)  # 对角线为1
        
        # 簇内高关联 (0.75-0.9)
        for cluster_id, indices in clusters.items():
            for i in indices:
                for j in indices:
                    if i != j:
                        C[i, j] = 0.8 + np.random.normal(0, 0.06)
                        C[i, j] = np.clip(C[i, j], 0.7, 0.95)
        
        # 簇间低关联 (0.1-0.4) 
        for c1_id, indices1 in clusters.items():
            for c2_id, indices2 in clusters.items():
                if c1_id != c2_id:
                    for i in indices1:
                        for j in indices2:
                            C[i, j] = 0.25 + np.random.normal(0, 0.05)
                            C[i, j] = np.clip(C[i, j], 0.1, 0.4)
        
        # 左右对称增强
        for i in range(6):
            l_idx, r_idx = i, i + 6
            if C[l_idx, r_idx] < 0.55:
                symmetric_val = 0.65 + np.random.normal(0, 0.04)
                C[l_idx, r_idx] = C[r_idx, l_idx] = symmetric_val
        
        # 层次聚类
        distance_matrix = 1 - C
        linkage_matrix = linkage(distance_matrix[np.triu_indices(12, k=1)], method='average')
        
        # 计算聚类统计
        cluster_labels = np.zeros(12, dtype=int)
        for cluster_id, indices in clusters.items():
            for idx in indices:
                cluster_labels[idx] = cluster_id
                
        cluster_stats = []
        for cluster_id, indices in clusters.items():
            size = len(indices)
            if size > 1:
                cluster_C = C[np.ix_(indices, indices)]
                jaccard_vals = cluster_C[np.triu_indices(size, k=1)]
                jaccard_mean = np.mean(jaccard_vals)
                jaccard_std = np.std(jaccard_vals)
            else:
                jaccard_mean = jaccard_std = 0.0
                
            # 简化轮廓系数
            intra_mean = np.mean([C[i, j] for i in indices for j in indices if i != j]) if size > 1 else 1.0
            inter_mean = np.mean([C[i, j] for i in indices for j in range(12) if cluster_labels[j] != cluster_id])
            silhouette = (1 - intra_mean - inter_mean) / max(1 - intra_mean, inter_mean, 0.001)
            
            cluster_stats.append((cluster_id, size, jaccard_mean, jaccard_std, silhouette))
        
        # 模拟时间活跃度 
        time_range = np.arange(220, 256)
        window_centers = np.arange(222, 254)  # Δt=5的窗口中心
        
        cluster_activities = {}
        for cluster_id in [1, 2, 3]:
            activities = []
            for t in window_centers:
                if 225 <= t <= 235:
                    base = 0.85 + 0.1 * np.sin((t-225)/10 * np.pi)
                elif 235 < t <= 245:
                    base = 0.9 + np.random.normal(0, 0.03) 
                elif 245 < t <= 250:
                    base = 0.9 - (t-245)/5 * 0.25
                else:
                    base = 0.25 + np.random.normal(0, 0.04)
                
                # 簇特异性
                if cluster_id == 1:
                    activity = base * 1.05
                elif cluster_id == 2: 
                    activity = base
                else:
                    activity = base * 0.85
                    
                activities.append(np.clip(activity, 0, 1))
            cluster_activities[cluster_id] = activities
        
        # 绘图
        fig = plt.figure(figsize=(12, 10))
        gs = fig.add_gridspec(4, 4, height_ratios=[0.8, 3.0, 0.8, 0.2], 
                             width_ratios=[1.0, 3.2, 0.3, 0.5], hspace=0.06, wspace=0.06)
        
        # 上方树状图
        ax_dendro_top = fig.add_subplot(gs[0, 1])
        dendro = dendrogram(linkage_matrix, ax=ax_dendro_top, orientation='top', no_labels=True,
                           color_threshold=0, above_threshold_color='#1f77b4', linewidth=2)
        ax_dendro_top.axis('off')
        
        # 左侧树状图  
        ax_dendro_left = fig.add_subplot(gs[1, 0])
        dendrogram(linkage_matrix, ax=ax_dendro_left, orientation='left', no_labels=True,
                  color_threshold=0, above_threshold_color='#1f77b4', linewidth=2)
        ax_dendro_left.axis('off')
        
        # 主热图 - 使用mako色谱
        ax_heatmap = fig.add_subplot(gs[1, 1])
        dendro_order = dendro['leaves']
        C_reordered = C[np.ix_(dendro_order, dendro_order)]
        reordered_labels = [cell_labels[i] for i in dendro_order]
        
        im = ax_heatmap.imshow(C_reordered, cmap='mako', vmin=0, vmax=1, aspect='equal')
        ax_heatmap.set_xticks(range(12))
        ax_heatmap.set_yticks(range(12))
        ax_heatmap.set_xticklabels(reordered_labels, rotation=45, ha='right', fontsize=self.tick_label_size)
        ax_heatmap.set_yticklabels(reordered_labels, fontsize=self.tick_label_size)
        
        # 簇边界白色方框
        reordered_clusters = [cluster_labels[i] for i in dendro_order]
        boundaries = []
        current_cluster = reordered_clusters[0]
        for i, cluster in enumerate(reordered_clusters[1:], 1):
            if cluster != current_cluster:
                boundaries.append(i - 0.5)
                current_cluster = cluster
                
        for boundary in boundaries:
            ax_heatmap.axhline(y=boundary, color='white', linewidth=3)
            ax_heatmap.axvline(x=boundary, color='white', linewidth=3)
        
        # 颜色条
        ax_cbar = fig.add_subplot(gs[1, 2])
        cbar = plt.colorbar(im, cax=ax_cbar)
        cbar.set_label('Co-association', fontsize=self.colorbar_size, weight='bold')
        cbar.ax.tick_params(labelsize=self.colorbar_size-1)
        
        # 侧注统计
        ax_stats = fig.add_subplot(gs[1, 3])
        ax_stats.axis('off')
        
        colors = ['#d62728', '#ff7f0e', '#2ca02c']
        y_pos = [0.85, 0.55, 0.25]
        
        for i, (cid, size, j_mean, j_std, sil) in enumerate(cluster_stats):
            stats_text = f"C{cid}\\nsize: {size}\\nJ: {j_mean:.3f}±{j_std:.3f}\\nS: {sil:.3f}"
            ax_stats.text(0.05, y_pos[i], stats_text, fontsize=self.legend_size,
                         transform=ax_stats.transAxes, va='top',
                         bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i], alpha=0.2))
        
        # 底部活跃度条带
        ax_activity = fig.add_subplot(gs[2, 1])
        
        for i, (cluster_id, activities) in enumerate(cluster_activities.items()):
            ax_activity.fill_between(window_centers, activities, alpha=0.6, color=colors[i], 
                                   label=f'C{cluster_id}')
            ax_activity.plot(window_centers, activities, color=colors[i], linewidth=2.5)
        
        ax_activity.set_xlabel('Time (minutes)', fontsize=self.axis_label_size, weight='bold') 
        ax_activity.set_ylabel('Activity', fontsize=self.tick_label_size)
        ax_activity.tick_params(labelsize=self.tick_label_size-1)
        ax_activity.legend(fontsize=self.legend_size-1, ncol=3)
        ax_activity.grid(True, alpha=0.3)
        ax_activity.set_ylim(0, 1.1)
        
        # 标题和解读
        fig.suptitle('Fig.1 背侧细胞一致性共关联矩阵：稳定同簇识别', 
                    fontsize=self.axis_label_size+2, weight='bold', y=0.96)
        
        interpretation = ('显示哪些细胞稳定同簇：C1(前端4细胞)、C2(中段6细胞)、C3(后端2细胞)\\n'
                         '聚类可信度：高稳定性(J>0.7) + 良好轮廓系数')
        fig.text(0.5, 0.02, interpretation, ha='center', fontsize=self.legend_size, style='italic',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.savefig(self.output_dir / 'Fig1_Dorsal_Consensus_Matrix.png', dpi=600, bbox_inches='tight')
        plt.close()
        print("✓ Fig.1 背侧一致性共关联矩阵已生成")
        
    def create_fig2_intestinal_consensus(self):
        """Fig.2 肠原基一致性共关联矩阵 (20×20)"""
        
        # E谱系细胞名称
        e_cells = ["int1DL", "int1VL", "int1DR", "int1VR"] + \
                  [f"int{i}L" for i in range(2, 10)] + [f"int{i}R" for i in range(2, 10)]
        
        # 构建20x20共关联矩阵
        np.random.seed(123)
        C = np.eye(20)
        
        # 定义肠原基簇（基于Ring和L/R）
        clusters = {
            1: [0, 1, 2, 3],                    # Ring1 (int1DL,VL,DR,VR)
            2: [4, 5, 6, 7, 8, 9],             # Ring2-4 左右
            3: [10, 11, 12, 13, 14, 15],       # Ring5-7 左右  
            4: [16, 17, 18, 19]                # Ring8-9 左右
        }
        
        # 簇内高关联
        for cluster_id, indices in clusters.items():
            for i in indices:
                for j in indices:
                    if i != j:
                        if cluster_id == 1:  # Ring1 最紧密
                            val = 0.85 + np.random.normal(0, 0.05)
                        else:
                            val = 0.75 + np.random.normal(0, 0.06)
                        C[i, j] = np.clip(val, 0.65, 0.95)
        
        # 簇间关联（Ring之间有梯度）
        for c1, indices1 in clusters.items():
            for c2, indices2 in clusters.items():
                if c1 != c2:
                    ring_dist = abs(c1 - c2)
                    base_corr = 0.4 - ring_dist * 0.08  # 距离越远关联越弱
                    for i in indices1:
                        for j in indices2:
                            C[i, j] = base_corr + np.random.normal(0, 0.04)
                            C[i, j] = np.clip(C[i, j], 0.05, 0.5)
        
        # 左右配对增强
        for i in range(4, 20, 2):  # int2L/R, int3L/R等配对
            if i+1 < 20:
                pair_strength = 0.7 + np.random.normal(0, 0.03)
                C[i, i+1] = C[i+1, i] = pair_strength
        
        # 层次聚类
        distance_matrix = 1 - C
        linkage_matrix = linkage(distance_matrix[np.triu_indices(20, k=1)], method='average')
        
        # Ring和L/R注释
        ring_annotations = ['R1']*4 + ['R2']*2 + ['R3']*2 + ['R4']*2 + ['R5']*2 + ['R6']*2 + ['R7']*2 + ['R8']*2 + ['R9']*2
        side_annotations = ['DL','VL','DR','VR'] + ['L','R']*8
        
        # 绘图
        fig = plt.figure(figsize=(13, 10))
        gs = fig.add_gridspec(5, 5, height_ratios=[0.6, 3.0, 0.3, 0.6, 0.2], 
                             width_ratios=[0.3, 0.3, 3.2, 0.3, 0.4], hspace=0.06, wspace=0.06)
        
        # 上方树状图
        ax_dendro_top = fig.add_subplot(gs[0, 2])
        dendro = dendrogram(linkage_matrix, ax=ax_dendro_top, orientation='top', no_labels=True,
                           color_threshold=0, above_threshold_color='#2E8B57', linewidth=1.8)
        ax_dendro_top.axis('off')
        
        # 主热图
        ax_heatmap = fig.add_subplot(gs[1, 2])
        dendro_order = dendro['leaves']
        C_reordered = C[np.ix_(dendro_order, dendro_order)]
        reordered_labels = [e_cells[i] for i in dendro_order]
        
        im = ax_heatmap.imshow(C_reordered, cmap='viridis', vmin=0, vmax=1, aspect='equal')
        ax_heatmap.set_xticks(range(20))
        ax_heatmap.set_yticks(range(20))
        ax_heatmap.set_xticklabels(reordered_labels, rotation=45, ha='right', fontsize=self.tick_label_size-2)
        ax_heatmap.set_yticklabels(reordered_labels, fontsize=self.tick_label_size-2)
        
        # 簇边界
        cluster_labels = np.zeros(20, dtype=int)
        for cluster_id, indices in clusters.items():
            for idx in indices:
                cluster_labels[idx] = cluster_id
                
        reordered_clusters = [cluster_labels[i] for i in dendro_order]
        boundaries = []
        current = reordered_clusters[0]
        for i, cluster in enumerate(reordered_clusters[1:], 1):
            if cluster != current:
                boundaries.append(i - 0.5)
                current = cluster
                
        for boundary in boundaries:
            ax_heatmap.axhline(y=boundary, color='white', linewidth=2.5)
            ax_heatmap.axvline(x=boundary, color='white', linewidth=2.5)
        
        # Ring侧注 
        ax_ring = fig.add_subplot(gs[1, 1])
        reordered_rings = [ring_annotations[i] for i in dendro_order]
        ring_colors = plt.cm.Set3(np.linspace(0, 1, 9))
        
        for i, ring in enumerate(reordered_rings):
            ring_num = int(ring[1]) - 1  # R1->0, R2->1...
            color = ring_colors[ring_num]
            ax_ring.barh(i, 1, color=color, alpha=0.8)
            
        ax_ring.set_xlim(0, 1)
        ax_ring.set_ylim(-0.5, 19.5)
        ax_ring.set_ylabel('Ring', fontsize=self.tick_label_size, rotation=0, ha='right')
        ax_ring.set_xticks([])
        ax_ring.set_yticks([])
        ax_ring.invert_yaxis()
        
        # L/R侧注
        ax_side = fig.add_subplot(gs[1, 0])
        reordered_sides = [side_annotations[i] for i in dendro_order]
        side_colors = {'L':'lightblue', 'R':'lightcoral', 'DL':'blue', 'VL':'cyan', 'DR':'red', 'VR':'orange'}
        
        for i, side in enumerate(reordered_sides):
            color = side_colors.get(side, 'gray')
            ax_side.barh(i, 1, color=color, alpha=0.8)
            
        ax_side.set_xlim(0, 1)
        ax_side.set_ylim(-0.5, 19.5) 
        ax_side.set_ylabel('Side', fontsize=self.tick_label_size, rotation=0, ha='right')
        ax_side.set_xticks([])
        ax_side.set_yticks([])
        ax_side.invert_yaxis()
        
        # 颜色条
        ax_cbar = fig.add_subplot(gs[1, 3])
        cbar = plt.colorbar(im, cax=ax_cbar)
        cbar.set_label('Co-association', fontsize=self.colorbar_size, weight='bold')
        cbar.ax.tick_params(labelsize=self.colorbar_size-1)
        
        # 聚类统计
        ax_stats = fig.add_subplot(gs[1, 4])
        ax_stats.axis('off')
        
        for i, (cid, indices) in enumerate(clusters.items()):
            size = len(indices)
            if size > 1:
                cluster_C = C[np.ix_(indices, indices)]
                jac_vals = cluster_C[np.triu_indices(size, k=1)]
                jac_mean, jac_std = np.mean(jac_vals), np.std(jac_vals)
            else:
                jac_mean = jac_std = 0
            
            stats_text = f"C{cid}\\nn={size}\\nJ={jac_mean:.2f}±{jac_std:.2f}"
            ax_stats.text(0.05, 0.9-i*0.22, stats_text, fontsize=self.legend_size-1,
                         transform=ax_stats.transAxes, va='top')
        
        # 底部阶段注记
        ax_stages = fig.add_subplot(gs[2, 2])
        stage_times = [360, 370, 380, 390]  # 肠原基发育阶段
        stage_labels = ['内化', '增殖', '重组', '管腔']
        stage_colors = ['red', 'orange', 'green', 'purple']
        
        for i, (time, label, color) in enumerate(zip(stage_times, stage_labels, stage_colors)):
            x_pos = i * 5  # 均匀分布
            ax_stages.axvline(x=x_pos, color=color, linestyle='--', alpha=0.8, linewidth=2)
            ax_stages.text(x_pos, 0.5, label, rotation=90, ha='center', va='center',
                          fontsize=self.legend_size, color=color, weight='bold')
        
        ax_stages.set_xlim(-1, 16)
        ax_stages.set_ylim(0, 1)
        ax_stages.set_xlabel('生物学阶段', fontsize=self.tick_label_size)
        ax_stages.set_xticks([])
        ax_stages.set_yticks([])
        
        # 标题和解读
        fig.suptitle('Fig.2 肠原基E谱系一致性共关联矩阵：环拓扑与稳定簇', 
                    fontsize=self.axis_label_size+2, weight='bold', y=0.96)
        
        interpretation = ('肠原基形成过程中稳定co-clusters：与Ring1-9环状拓扑和L/R对称性一致\n'
                         'C1(Ring1,4细胞)、C2-C4(Ring2-9,梯度关联)')
        fig.text(0.5, 0.02, interpretation, ha='center', fontsize=self.legend_size, style='italic',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))\n        \n        plt.savefig(self.output_dir / 'Fig2_Intestinal_Consensus_Matrix.png', dpi=600, bbox_inches='tight')\n        plt.close()\n        print(\"✓ Fig.2 肠原基一致性共关联矩阵已生成\")\n        \n    def create_fig3_cluster_meaning_panels(self):\n        \"\"\"Fig.3 四联图A-D：簇意义解析\"\"\"\n        \n        # 模拟肠原基数据优先\n        np.random.seed(456)\n        time_range = np.arange(350, 401)  # 肠原基时间\n        n_cells = 20\n        \n        # 定义4个簇\n        clusters = {1: list(range(4)), 2: list(range(4, 10)), 3: list(range(10, 16)), 4: list(range(16, 20))}\n        cluster_labels = np.zeros(n_cells, dtype=int)\n        for cid, indices in clusters.items():\n            for idx in indices:\n                cluster_labels[idx] = cid\n        \n        # 生成数据\n        # A. 簇活跃度随时间\n        window_centers = np.arange(352, 399, 2)  # 滑动窗口中心\n        cluster_activities = {}\n        \n        for cid in [1, 2, 3, 4]:\n            activities = []\n            for t in window_centers:\n                if 355 <= t <= 365:  # 内化期\n                    base = 0.8 + 0.1 * np.sin((t-355)/10 * np.pi)\n                elif 365 < t <= 375:  # 增殖期\n                    base = 0.9 + np.random.normal(0, 0.04)\n                elif 375 < t <= 385:  # 重组期\n                    base = 0.95 + np.random.normal(0, 0.03)\n                elif 385 < t <= 395:  # 管腔期\n                    base = 0.85 - (t-385)/10 * 0.2\n                else:\n                    base = 0.3 + np.random.normal(0, 0.05)\n                \n                # 簇差异\n                if cid == 1:  # Ring1早期最活跃\n                    activity = base * 1.1\n                elif cid == 2:  # Ring2-4中等\n                    activity = base * 1.0\n                elif cid == 3:  # Ring5-7\n                    activity = base * 0.95\n                else:  # Ring8-9后期\n                    activity = base * 0.85\n                    \n                activities.append(np.clip(activity, 0, 1))\n            cluster_activities[cid] = activities\n        \n        # B. 运动学数据（负Z速度）\n        velocity_data = []\n        velocity_clusters = []\n        \n        for cid in [1, 2, 3, 4]:\n            n_samples = len(clusters[cid]) * len(time_range)  # 每个细胞每个时间点\n            \n            if cid == 1:  # Ring1内化最强\n                velocities = np.random.normal(-8.0, 1.5, n_samples)\n            elif cid == 2:\n                velocities = np.random.normal(-5.5, 1.2, n_samples)\n            elif cid == 3:\n                velocities = np.random.normal(-4.0, 1.0, n_samples)\n            else:  # Ring8-9内化最弱\n                velocities = np.random.normal(-2.5, 0.8, n_samples)\n                \n            velocity_data.extend(velocities)\n            velocity_clusters.extend([cid] * n_samples)\n        \n        # C. 空间收敛数据（到管轴距离）\n        spatial_distances = {}\n        \n        for cid in [1, 2, 3, 4]:\n            distances = []\n            for t in time_range:\n                if cid == 1:  # Ring1收敛最快\n                    base_dist = 25 - (t - 350) * 0.45\n                elif cid == 2:\n                    base_dist = 30 - (t - 350) * 0.35\n                elif cid == 3:\n                    base_dist = 35 - (t - 350) * 0.28\n                else:  # Ring8-9收敛最慢\n                    base_dist = 40 - (t - 350) * 0.20\n                    \n                distances.append(max(base_dist + np.random.normal(0, 2), 3))\n            spatial_distances[cid] = distances\n        \n        # D. 特征数据（6个几何特征）\n        feature_names = ['Z-axis Velocity', 'Apical Surface Area', 'Cell Volume Change',\n                        'Radial Distance', 'Cell Sphericity', 'Neighbor Contact Number']\n        \n        feature_data = {}\n        for fname in feature_names:\n            values = []\n            for cell_idx in range(n_cells):\n                cid = cluster_labels[cell_idx]\n                \n                if fname == 'Z-axis Velocity':\n                    if cid == 1: val = -8.0 + np.random.normal(0, 1.2)\n                    elif cid == 2: val = -5.5 + np.random.normal(0, 1.0)\n                    elif cid == 3: val = -4.0 + np.random.normal(0, 0.8)\n                    else: val = -2.5 + np.random.normal(0, 0.6)\n                elif fname == 'Apical Surface Area':\n                    if cid == 1: val = 35 + np.random.normal(0, 4)  # 更小（收缩）\n                    else: val = 55 + np.random.normal(0, 6)\n                elif fname == 'Cell Volume Change':\n                    if cid == 1: val = -0.25 + np.random.normal(0, 0.04)  # 体积减小\n                    else: val = -0.08 + np.random.normal(0, 0.06)\n                elif fname == 'Radial Distance':\n                    if cid == 1: val = 12 + np.random.normal(0, 2)\n                    else: val = 22 + np.random.normal(0, 3)\n                elif fname == 'Cell Sphericity':\n                    if cid == 1: val = 0.55 + np.random.normal(0, 0.06)  # 更不规则\n                    else: val = 0.75 + np.random.normal(0, 0.08)\n                else:  # Neighbor Contact Number\n                    if cid == 2: val = 7.5 + np.random.normal(0, 1.0)  # 中间环接触多\n                    else: val = 5.8 + np.random.normal(0, 0.8)\n                    \n                values.append(val)\n            feature_data[fname] = np.array(values)\n        \n        # 绘制四联图\n        fig, ((ax_a, ax_b), (ax_c, ax_d)) = plt.subplots(2, 2, figsize=(16, 12))\n        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']\n        \n        # A. Cluster dynamics over time\n        for i, (cid, activities) in enumerate(cluster_activities.items()):\n            activities = np.array(activities)\n            ax_a.plot(window_centers, activities, color=colors[i], linewidth=2.5, \n                     label=f'Cluster {cid}', marker='o', markersize=3)\n            # SEM误差带（模拟）\n            sem = activities * 0.05  # 5%的SEM\n            ax_a.fill_between(window_centers, activities-sem, activities+sem, \n                            color=colors[i], alpha=0.2)\n        \n        # 阶段边界虚线\n        stage_times = [355, 365, 375, 385, 395]\n        stage_labels = ['', '内化', '增殖', '重组', '管腔']\n        for i, (t, label) in enumerate(zip(stage_times, stage_labels)):\n            if 350 <= t <= 400:\n                ax_a.axvline(t, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)\n                if label:\n                    ax_a.text(t, 0.95, label, rotation=90, ha='right', va='top', \n                             fontsize=self.legend_size-1, color='gray')\n        \n        ax_a.set_xlabel('Time (minutes)', fontsize=self.axis_label_size)\n        ax_a.set_ylabel('Cluster Activity (mean±SEM)', fontsize=self.axis_label_size)\n        ax_a.set_title('(A) Cluster dynamics over time', fontsize=self.axis_label_size, weight='bold')\n        ax_a.legend(fontsize=self.legend_size)\n        ax_a.grid(True, alpha=0.3)\n        ax_a.set_ylim(0, 1.1)\n        \n        # B. Kinematics per cluster (小提琴图)\n        cluster_velocity_data = []\n        for cid in [1, 2, 3, 4]:\n            cluster_mask = np.array(velocity_clusters) == cid\n            cluster_velocity_data.append(np.array(velocity_data)[cluster_mask])\n        \n        parts = ax_b.violinplot(cluster_velocity_data, positions=[1, 2, 3, 4], widths=0.6, showmeans=True)\n        for i, pc in enumerate(parts['bodies']):\n            pc.set_facecolor(colors[i])\n            pc.set_alpha(0.7)\n        \n        # Kruskal-Wallis检验\n        try:\n            h_stat, p_val = stats.kruskal(*cluster_velocity_data)\n            sig_text = f'Kruskal-Wallis: H={h_stat:.2f}, p={p_val:.3f}'\n            if p_val < 0.05:\n                sig_text += '*'\n            ax_b.text(0.02, 0.98, sig_text, transform=ax_b.transAxes, \n                     fontsize=self.legend_size-1, va='top')\n        except:\n            pass\n        \n        ax_b.set_xlabel('Cluster', fontsize=self.axis_label_size)\n        ax_b.set_ylabel('Internalization Velocity (μm/min)', fontsize=self.axis_label_size)\n        ax_b.set_title('(B) Kinematics per cluster', fontsize=self.axis_label_size, weight='bold')\n        ax_b.set_xticks([1, 2, 3, 4])\n        ax_b.grid(True, alpha=0.3)\n        \n        # C. Spatial convergence\n        for i, (cid, distances) in enumerate(spatial_distances.items()):\n            distances = np.array(distances)\n            # 中位数±IQR（简化为均值±std）\n            ax_c.plot(time_range, distances, color=colors[i], linewidth=2.5, \n                     label=f'C{cid}', alpha=0.8)\n            \n            # 线性回归斜率\n            slope, intercept, r_val, p_val, std_err = stats.linregress(time_range, distances)\n            ci_95 = 1.96 * std_err\n            \n            # 右上角标注斜率\n            ax_c.text(0.98, 0.98-i*0.08, f'C{cid}: {slope:.3f}±{ci_95:.3f}', \n                     transform=ax_c.transAxes, fontsize=self.legend_size-1, \n                     ha='right', va='top', color=colors[i])\n        \n        ax_c.set_xlabel('Time (minutes)', fontsize=self.axis_label_size)\n        ax_c.set_ylabel('Distance from Gut Axis (μm)', fontsize=self.axis_label_size)\n        ax_c.set_title('(C) Spatial meaning: convergence', fontsize=self.axis_label_size, weight='bold')\n        ax_c.legend(fontsize=self.legend_size)\n        ax_c.grid(True, alpha=0.3)\n        \n        # D. Feature enrichment (森林图)\n        effect_sizes = []\n        feature_cluster_labels = []\n        \n        for fname in feature_names:\n            values = feature_data[fname]\n            for cid in [1, 2, 3, 4]:\n                cluster_mask = cluster_labels == cid\n                cluster_data = values[cluster_mask]\n                other_data = values[~cluster_mask]\n                \n                if len(cluster_data) > 0 and len(other_data) > 0:\n                    # Cohen's d\n                    pooled_std = np.sqrt(((len(cluster_data)-1)*np.var(cluster_data, ddof=1) +\n                                        (len(other_data)-1)*np.var(other_data, ddof=1)) /\n                                       (len(cluster_data) + len(other_data) - 2))\n                    cohens_d = (np.mean(cluster_data) - np.mean(other_data)) / pooled_std if pooled_std > 0 else 0\n                    \n                    # 95% CI\n                    se = np.sqrt((len(cluster_data) + len(other_data)) / (len(cluster_data) * len(other_data)) +\n                               cohens_d**2 / (2*(len(cluster_data) + len(other_data))))\n                    ci_lower = cohens_d - 1.96 * se\n                    ci_upper = cohens_d + 1.96 * se\n                    \n                    effect_sizes.append((cohens_d, ci_lower, ci_upper))\n                    feature_cluster_labels.append(f'{fname} (C{cid})')\n        \n        # 按绝对效应量排序\n        sorted_indices = sorted(range(len(effect_sizes)), key=lambda i: abs(effect_sizes[i][0]), reverse=True)\n        \n        # 绘制森林图（取前12个）\n        y_positions = range(min(12, len(sorted_indices)))\n        for i, idx in enumerate(sorted_indices[:12]):\n            d, ci_low, ci_high = effect_sizes[idx]\n            label = feature_cluster_labels[idx]\n            \n            color = colors[i % 4]\n            ax_d.scatter(d, i, color=color, s=80, alpha=0.8, zorder=3)\n            ax_d.plot([ci_low, ci_high], [i, i], color=color, linewidth=2.5, alpha=0.7, zorder=2)\n            \n            # 标签\n            ax_d.text(-0.02, i, label, ha='right', va='center', \n                     transform=ax_d.get_yaxis_transform(), fontsize=self.legend_size-2)\n        \n        ax_d.axvline(0, color='black', linestyle='--', alpha=0.7, linewidth=1.5)\n        ax_d.set_xlabel('Effect Size (Cohen\\'s d)', fontsize=self.axis_label_size)\n        ax_d.set_title('(D) Feature enrichment', fontsize=self.axis_label_size, weight='bold')\n        ax_d.grid(True, alpha=0.3)\n        \n        # 方向解释\n        ax_d.text(0.5, -0.12, '← 簇内较少 | 簇内较多 →', ha='center', \n                 transform=ax_d.transAxes, fontsize=self.legend_size-1, style='italic')\n        \n        # 总标题\n        fig.suptitle('Fig.3 簇生物学意义四联图：动力学-空间-特征综合分析', \n                    fontsize=self.axis_label_size+2, weight='bold', y=0.96)\n        \n        plt.tight_layout()\n        plt.savefig(self.output_dir / 'Fig3_Cluster_Meaning_Panels.png', dpi=600, bbox_inches='tight')\n        plt.close()\n        print(\"✓ Fig.3 四联图簇意义分析已生成\")\n\nif __name__ == \"__main__\":\n    generator = TestPlotGenerator()\n    \n    print(\"开始生成三张重新设计的图...\")\n    generator.create_fig1_dorsal_consensus()\n    generator.create_fig2_intestinal_consensus()\n    generator.create_fig3_cluster_meaning_panels()\n    \n    print(\"\\n✅ 所有图表已生成完成！\")\n    print(f\"输出目录: {generator.output_dir.absolute()}\")\n    print(\"\\n三张图的特点：\")\n    print(\"Fig.1: 12×12背侧一致性共关联矩阵，突出稳定簇识别与可信度\")\n    print(\"Fig.2: 20×20肠原基共关联矩阵，展示Ring拓扑与L/R对称性\")\n    print(\"Fig.3: A-D四联图，全面解析簇的生物学意义（动力学+空间+特征）\")