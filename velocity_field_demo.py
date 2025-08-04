#!/usr/bin/env python3
"""
速度场分析Demo - 展示背侧插入细胞的速度分布
这个demo使用模拟数据演示速度场分析的可视化效果
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class VelocityFieldDemo:
    def __init__(self):
        self.setup_demo_data()
    
    def setup_demo_data(self):
        """设置模拟的细胞速度数据"""
        # 时间范围：220-250分钟，每2分钟一个时间点
        self.time_bins = np.arange(220, 250, 2)
        
        # 模拟12个背侧插入细胞的速度数据
        self.n_cells = 12
        self.cell_ids = [f"DC{i+1:02d}" for i in range(self.n_cells)]
        
        # 为每个时间点生成速度数据
        self.velocity_data = self.generate_realistic_velocity_data()
    
    def generate_realistic_velocity_data(self):
        """生成符合生物学规律的速度数据"""
        velocity_data = {}
        
        for i, time_point in enumerate(self.time_bins):
            velocities = []
            
            # 根据时间模拟不同的速度分布模式
            if 220 <= time_point < 230:
                # 早期：低速度，小变异
                base_velocity = 0.5
                noise_level = 0.2
            elif 230 <= time_point < 240:
                # 中期：高速度，大变异（活跃期）
                base_velocity = 2.0
                noise_level = 0.8
            else:
                # 晚期：中等速度，中等变异
                base_velocity = 1.2
                noise_level = 0.4
            
            # 为每个细胞生成速度值
            for cell_idx in range(self.n_cells):
                # 添加细胞间的个体差异
                cell_factor = 0.8 + 0.4 * np.random.random()
                velocity = base_velocity * cell_factor + np.random.normal(0, noise_level)
                
                # 确保速度不为负值（取绝对值）
                velocity = abs(velocity)
                velocities.append(velocity)
            
            velocity_data[time_point] = velocities
        
        return velocity_data
    
    def plot_velocity_field(self, save_path="velocity_field_demo.png"):
        """生成速度场分析图"""
        plt.figure(figsize=(14, 8))
        
        # 准备箱线图数据
        velocities_for_boxplot = []
        positions = []
        
        for time_point in self.time_bins:
            if time_point in self.velocity_data:
                velocities_for_boxplot.append(self.velocity_data[time_point])
                positions.append(time_point)
        
        # 创建箱线图
        box_plot = plt.boxplot(
            velocities_for_boxplot,
            positions=positions,
            widths=1.5,
            patch_artist=True,
            showfliers=True,
            flierprops=dict(marker='o', markerfacecolor='red', markersize=5, alpha=0.6)
        )
        
        # 美化箱线图
        colors = plt.cm.viridis(np.linspace(0, 1, len(box_plot['boxes'])))
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # 添加中位数标注
        for i, time_point in enumerate(positions):
            median_vel = np.median(self.velocity_data[time_point])
            plt.text(time_point, median_vel + 0.1, f'{median_vel:.1f}', 
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # 设置图表样式
        plt.xlabel('Time (minutes post-fertilization)', fontsize=12, fontweight='bold')
        plt.ylabel('Midline Crossing Velocity (μm/min)', fontsize=12, fontweight='bold')
        plt.title('Velocity Field Analysis\nDorsal Intercalation Cells (Demo)', 
                 fontsize=14, fontweight='bold', pad=20)
        
        # 添加背景网格
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # 标注关键时期
        plt.axvspan(230, 240, alpha=0.2, color='yellow', label='Active Intercalation Period')
        
        # 添加图例和注释
        plt.legend(loc='upper right')
        
        # 添加统计信息文本框
        stats_text = f"Cells analyzed: {self.n_cells}\nTime points: {len(self.time_bins)}\nTotal measurements: {sum(len(v) for v in self.velocity_data.values())}"
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path
    
    def plot_velocity_heatmap(self, save_path="velocity_heatmap_demo.png"):
        """生成速度热图，展示每个细胞随时间的速度变化"""
        plt.figure(figsize=(12, 8))
        
        # 准备热图数据矩阵
        velocity_matrix = np.zeros((self.n_cells, len(self.time_bins)))
        
        for j, time_point in enumerate(self.time_bins):
            velocities = self.velocity_data[time_point]
            for i in range(self.n_cells):
                velocity_matrix[i, j] = velocities[i]
        
        # 创建热图
        sns.heatmap(
            velocity_matrix,
            xticklabels=[f'{t}' for t in self.time_bins],
            yticklabels=self.cell_ids,
            cmap='RdYlBu_r',
            cbar_kws={'label': 'Velocity (μm/min)'},
            annot=True,
            fmt='.1f',
            annot_kws={'size': 8}
        )
        
        plt.xlabel('Time (minutes post-fertilization)', fontsize=12, fontweight='bold')
        plt.ylabel('Dorsal Cell ID', fontsize=12, fontweight='bold')
        plt.title('Individual Cell Velocity Heatmap\nDorsal Intercalation Analysis (Demo)', 
                 fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path
    
    def plot_velocity_trends(self, save_path="velocity_trends_demo.png"):
        """绘制速度趋势图，显示平均速度随时间的变化"""
        plt.figure(figsize=(12, 8))
        
        # 计算每个时间点的统计量
        mean_velocities = []
        std_velocities = []
        max_velocities = []
        min_velocities = []
        
        for time_point in self.time_bins:
            velocities = self.velocity_data[time_point]
            mean_velocities.append(np.mean(velocities))
            std_velocities.append(np.std(velocities))
            max_velocities.append(np.max(velocities))
            min_velocities.append(np.min(velocities))
        
        mean_velocities = np.array(mean_velocities)
        std_velocities = np.array(std_velocities)
        
        # 绘制均值线和置信区间
        plt.plot(self.time_bins, mean_velocities, 'b-', linewidth=3, label='Mean Velocity')
        plt.fill_between(self.time_bins, 
                        mean_velocities - std_velocities,
                        mean_velocities + std_velocities,
                        alpha=0.3, color='blue', label='±1 Standard Deviation')
        
        # 绘制最大值和最小值
        plt.plot(self.time_bins, max_velocities, 'r--', alpha=0.7, label='Maximum')
        plt.plot(self.time_bins, min_velocities, 'g--', alpha=0.7, label='Minimum')
        
        # 标注关键时期
        plt.axvspan(230, 240, alpha=0.2, color='yellow', label='Active Period')
        
        plt.xlabel('Time (minutes post-fertilization)', fontsize=12, fontweight='bold')
        plt.ylabel('Velocity (μm/min)', fontsize=12, fontweight='bold')
        plt.title('Velocity Trends Over Time\nDorsal Intercalation Analysis (Demo)', 
                 fontsize=14, fontweight='bold', pad=20)
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path
    
    def generate_all_plots(self, output_dir="dorsal_plots"):
        """生成所有速度分析图表"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        plots = {}
        
        print("生成速度场分析图...")
        plots['velocity_field'] = self.plot_velocity_field(output_dir / "demo_velocity_field.png")
        
        print("生成速度热图...")
        plots['velocity_heatmap'] = self.plot_velocity_heatmap(output_dir / "demo_velocity_heatmap.png")
        
        print("生成速度趋势图...")
        plots['velocity_trends'] = self.plot_velocity_trends(output_dir / "demo_velocity_trends.png")
        
        return plots
    
    def print_data_summary(self):
        """打印数据摘要"""
        print("\n=== 速度场分析Demo数据摘要 ===")
        print(f"分析细胞数量: {self.n_cells}")
        print(f"时间点数量: {len(self.time_bins)}")
        print(f"时间范围: {self.time_bins[0]}-{self.time_bins[-1]} 分钟")
        
        print("\n各时间点速度统计:")
        for time_point in self.time_bins:
            velocities = self.velocity_data[time_point]
            print(f"  {time_point}分钟: 均值={np.mean(velocities):.2f}, "
                  f"标准差={np.std(velocities):.2f}, "
                  f"范围=[{np.min(velocities):.2f}, {np.max(velocities):.2f}]")


def main():
    """主函数"""
    print("启动速度场分析Demo...")
    
    # 创建demo实例
    demo = VelocityFieldDemo()
    
    # 打印数据摘要
    demo.print_data_summary()
    
    # 生成所有图表
    plots = demo.generate_all_plots()
    
    print("\n=== 生成的图表 ===")
    for plot_type, path in plots.items():
        print(f"  {plot_type}: {path}")
    
    print("\nDemo完成！")
    
    # 返回demo实例供进一步分析
    return demo


if __name__ == "__main__":
    demo = main()
