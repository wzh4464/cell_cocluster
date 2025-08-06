#!/usr/bin/env python3
"""
统一demo调用脚本
生成7张demo图：
1. Demo1A_Dorsal_Left_Coclustering_Heatmap.png - 背侧嵌入左侧共聚类热图
2. Demo1B_Dorsal_Right_Coclustering_Heatmap.png - 背侧嵌入右侧共聚类热图
3. Demo2_Dorsal_Cell_Trajectories.png - 背侧细胞轨迹图
4. Demo4_Center_High_Coclustering_Heatmap.png - 时间递减共聚类热图（后半段整体下降）
5. Demo6_Intestinal_Velocity_Field.png - 肠原基速度场分析
6. Demo7A_Dorsal_Coclustering_Features_Pie.png - Dorsal intercalation geometrical features
7. Demo7B_Intestinal_Coclustering_Features_Pie.png - Intestinal morphogenesis geometrical features

使用方法:
    python run_all_demos.py                    # 默认字体大小
    python run_all_demos.py --font-scale 1.5   # 字体放大1.5倍
    python run_all_demos.py --font-scale 0.8   # 字体缩小到0.8倍
"""

from dorsal_intercalation_analysis import DorsalIntercalationAnalyzer, FontConfig
import time
import argparse
from pathlib import Path

def main():
    """运行所有demo生成6张图。"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='生成细胞共聚类demo图表')
    parser.add_argument('--font-scale', type=float, default=1.0,
                       help='字体缩放因子 (默认: 1.0, 示例: 1.5表示放大1.5倍)')
    parser.add_argument('--output-dir', type=str, default='demo_plots',
                       help='输出目录 (默认: demo_plots)')
    
    args = parser.parse_args()
    
    print("🔬 Cell Co-clustering Demo System")
    print("=" * 50)
    print(f"字体缩放因子: {args.font_scale}")
    print(f"输出目录: {args.output_dir}")
    print("正在生成7张demo图表...")
    print()
    
    start_time = time.time()
    
    # 创建字体配置
    font_config = FontConfig(scale_factor=args.font_scale)
    
    # 初始化分析器
    analyzer = DorsalIntercalationAnalyzer(font_config=font_config)
    
    # 生成所有demo图表
    plots = analyzer.generate_all_plots(output_dir=args.output_dir)
    
    end_time = time.time()
    
    print()
    print("🎉 Demo生成完成!")
    print(f"⏱️  总耗时: {end_time - start_time:.2f}秒")
    print()
    print("📊 生成的图表:")
    print("-" * 30)
    
    demo_descriptions = {
        "dorsal_left_coclustering": "🔥 背侧嵌入左侧共聚类热图",
        "dorsal_right_coclustering": "🔥 背侧嵌入右侧共聚类热图",
        "dorsal_trajectories": "🔄 背侧细胞轨迹分析", 
        "demo4_coclustering": "🎯 时间递减共聚类热图（后半段整体下降）",
        "intestinal_velocity": "⬇️  内化速度场分析",
        "dorsal_features_pie": "🥧 Dorsal Co-clustering Features Pie",
        "intestinal_features_pie": "🥧 Intestinal Co-clustering Features Pie"
    }
    
    for plot_name, file_path in plots.items():
        description = demo_descriptions.get(plot_name, plot_name)
        print(f"{description}: {file_path}")
    
    print()
    print("🔬 Demo特征说明:")
    print("📈 背侧嵌入(Dorsal Intercalation):")
    print("   - 220-250分钟发育时期")
    print("   - 细胞跨中线运动模式")
    print("   - 左右分群聚类行为")
    print()
    print("🎯 Demo4时间递减模式:")
    print("   - 所有细胞均匀高概率期(225-240分钟)")
    print("   - 后半段时间整体概率下降(240-255分钟)")
    print("   - 展示统一的时间性概率递减模式")
    print()
    print("🍼 肠原基形成(Intestinal Primordium):")
    print("   - 350-400分钟发育时期")
    print("   - E谱系20个细胞")
    print("   - 内化运动(负Z速度)")
    print("   - 单一细胞谱系同源接触")
    print()
    print("🥧 Co-clustering Feature Distribution Pies:")
    print("   - Quantitative local geometrical properties")
    print("   - Dorsal: Y-axis velocity(26%), Cell elongation(21%), Surface curvature(18%)")
    print("   - Intestinal: Z-axis velocity(28%), Apical surface area(24%), Volume change(19%)")
    print("   - Shows relative importance of measurable features in co-clustering")
    print()
    print("📏 字体大小:")
    print(f"   - 轴标签: {font_config.axis_label_size}pt")
    print(f"   - 刻度标签: {font_config.tick_label_size}pt") 
    print(f"   - 图例: {font_config.legend_size}pt")
    print(f"   - 色标: {font_config.colorbar_size}pt")

if __name__ == "__main__":
    main()