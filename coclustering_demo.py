#!/usr/bin/env python3
"""
Demo script to show what an ideal co-clustering probability heatmap should look like
for dorsal intercalation analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import ndimage
import warnings

warnings.filterwarnings("ignore")


def create_ideal_coclustering_demo():
    """Create demo of ideal co-clustering probability patterns for left/right dorsal cell groups."""

    # Time range: 220-255 minutes (discrete integer minutes)
    time_range = np.arange(220, 256)
    n_cells_per_side = 6  # 6 cells per side

    # Create probability matrices for left and right sides separately
    left_matrix = np.zeros((n_cells_per_side, len(time_range)))
    right_matrix = np.zeros((n_cells_per_side, len(time_range)))

    # 新的聚类概率时间段设定
    cluster_rise_start = 225
    cluster_rise_end = 230
    cluster_high_start = 230
    cluster_high_end = 254
    cluster_fall_start = 255
    
    for i in range(n_cells_per_side):
        for j, t in enumerate(time_range):
            if cluster_rise_start <= t <= cluster_rise_end:
                # 225-230分钟快速上升
                progress = (t - cluster_rise_start) / (cluster_rise_end - cluster_rise_start)
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


def plot_ideal_heatmaps():
    """Plot separate left and right co-clustering probability heatmaps."""
    left_matrix, right_matrix, time_range = create_ideal_coclustering_demo()

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
    return fig


def plot_comparison_bad_vs_good():
    """Show comparison between poor and ideal results for left side only."""
    left_matrix, _, time_range = create_ideal_coclustering_demo()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Bad example: random noise
    n_cells = 6
    bad_matrix = np.random.random((n_cells, len(time_range)))

    # Plot bad example
    sns.heatmap(
        bad_matrix,
        xticklabels=[str(t) if t % 5 == 0 else "" for t in time_range],
        yticklabels=[f"L{i+1:02d}" for i in range(n_cells)],
        cmap="RdBu_r",
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Probability"},
        ax=ax1,
    )
    ax1.set_xlabel("Time (minutes)")
    ax1.set_ylabel("Left Dorsal Cells")

    # Plot good example
    sns.heatmap(
        left_matrix,
        xticklabels=[str(t) if t % 5 == 0 else "" for t in time_range],
        yticklabels=[f"L{i+1:02d}" for i in range(n_cells)],
        cmap="RdBu_r",
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Probability"},
        ax=ax2,
    )
    ax2.set_xlabel("Time (minutes)")
    ax2.set_ylabel("Left Dorsal Cells")

    plt.tight_layout()
    return fig


def analyze_patterns():
    """Analyze what makes a good pattern."""
    left_matrix, right_matrix, time_range = create_ideal_coclustering_demo()

    print("理想共聚类热图的特征 (Characteristics of Ideal Co-clustering Heatmap):")
    print("=" * 70)

    # 1. Temporal clustering for left side
    print("1. 左侧细胞时间聚类模式 (Left Side Temporal Clustering):")
    for i in range(left_matrix.shape[0]):
        peak_time_idx = np.argmax(left_matrix[i, :])
        peak_time = time_range[peak_time_idx]
        peak_prob = left_matrix[i, peak_time_idx]
        print(f"   Cell L{i+1:02d}: Peak at {peak_time} min (prob={peak_prob:.2f})")

    print("\n2. 右侧细胞时间聚类模式 (Right Side Temporal Clustering):")
    for i in range(right_matrix.shape[0]):
        peak_time_idx = np.argmax(right_matrix[i, :])
        peak_time = time_range[peak_time_idx]
        peak_prob = right_matrix[i, peak_time_idx]
        print(f"   Cell R{i+1:02d}: Peak at {peak_time} min (prob={peak_prob:.2f})")

    print("\n3. 概率分布统计 (Probability Distribution Statistics):")
    print(
        f"   Left mean: {np.mean(left_matrix):.3f}, Right mean: {np.mean(right_matrix):.3f}"
    )
    print(
        f"   Left max: {np.max(left_matrix):.3f}, Right max: {np.max(right_matrix):.3f}"
    )

    print("\n4. 生物学意义 (Biological Significance):")
    print("   - 左右两群同时开始聚类(220-222分钟)")
    print("   - 峰值出现在226分钟左右")
    print("   - 232分钟后聚类活动减弱")
    print("   - 时间范围扩展到255分钟")
    print("   - 左右两群表现出相似的时间模式")


def main():
    """Generate demo plots showing ideal co-clustering patterns."""
    from pathlib import Path

    # Create output directory
    output_dir = Path("dorsal_plots")
    output_dir.mkdir(exist_ok=True)

    print("生成理想共聚类概率热图演示...")
    print("Generating ideal co-clustering probability heatmap demo...")

    # Generate separate left/right heatmaps
    fig1 = plot_ideal_heatmaps()
    fig1.savefig(
        output_dir / "ideal_coclustering_left_right.png", dpi=300, bbox_inches="tight"
    )
    print(f"✅ 保存左右分离热图: {output_dir}/ideal_coclustering_left_right.png")
    plt.close(fig1)

    # Generate comparison
    fig2 = plot_comparison_bad_vs_good()
    fig2.savefig(
        output_dir / "coclustering_comparison.png", dpi=300, bbox_inches="tight"
    )
    print(f"✅ 保存对比图: {output_dir}/coclustering_comparison.png")
    plt.close(fig2)

    # Analyze patterns
    analyze_patterns()

    print("\n" + "=" * 50)
    print("DEMO 完成! 关键特征:")
    print("1. 左右分离 - 两幅独立的热图")
    print("2. 同时聚类 - 220-222分钟开始，226分钟峰值")
    print("3. 时间范围 - 220-255分钟(整数)")
    print("4. 无标题 - 简洁的图表设计")
    print("5. 高对比度 - 使用viridis色彩方案")


if __name__ == "__main__":
    main()
