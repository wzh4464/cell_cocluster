# 细胞共聚类Demo系统

## 快速开始

运行以下命令生成全部7张demo图：

```bash
.venv/bin/python run_all_demos.py
```

## Demo内容

### 背侧嵌入 (Dorsal Intercalation) - 4张图

1. **Demo1A_Dorsal_Left_Coclustering_Heatmap.png**
   - 背侧左侧细胞共聚类概率热图（正方形）
   - 时间范围：220-255分钟
   - 左侧6个细胞的理想聚类模式

2. **Demo1B_Dorsal_Right_Coclustering_Heatmap.png**
   - 背侧右侧细胞共聚类概率热图（正方形）
   - 时间范围：220-255分钟
   - 右侧6个细胞的理想聚类模式

3. **Demo2_Dorsal_Cell_Trajectories.png**
   - 背侧细胞轨迹可视化
   - 显示细胞向中线汇聚的运动模式
   - 12个细胞（左右各6个）

4. **Demo3_Dorsal_Velocity_Field.png**
   - 背侧细胞速度场分析
   - violin plot展示跨中线速度分布
   - 突出225-240分钟活跃期

### 肠原基形成 (Intestinal Primordium Formation) - 3张图

5. **Demo4_Intestinal_Coclustering_Heatmap.png**
   - E谱系细胞共聚类热图
   - 时间范围：350-400分钟
   - 20个E谱系细胞的内化聚类

6. **Demo5_Intestinal_Trajectories.png**
   - E谱系细胞3D内化轨迹
   - 从表面向内部的运动轨迹
   - 展示管状器官形成过程

7. **Demo6_Intestinal_Velocity_Field.png**
   - 内化速度场分析
   - 负Z方向速度（向内运动）
   - violin plot显示内化动力学

## 核心特征

### 背侧嵌入特征
- **时期**：220-250分钟（受精后）
- **过程**：细胞跨中线的对称运动
- **聚类**：225-230分钟快速上升，230-254分钟高概率聚类
- **模式**：左右分群，同时聚类

### 肠原基形成特征  
- **时期**：350-400分钟（受精后）
- **谱系**：单一E谱系，20个细胞
- **运动**：表面向内的内化运动（负Z速度）
- **特征**：内化运动、顶-基底极性、同源细胞接触

## 文件结构

```
demo_plots/
├── Demo1A_Dorsal_Left_Coclustering_Heatmap.png  # 背侧左侧共聚类（正方形）
├── Demo1B_Dorsal_Right_Coclustering_Heatmap.png # 背侧右侧共聚类（正方形）
├── Demo2_Dorsal_Cell_Trajectories.png           # 背侧轨迹
├── Demo3_Dorsal_Velocity_Field.png              # 背侧速度场
├── Demo4_Intestinal_Coclustering_Heatmap.png    # 肠道共聚类
├── Demo5_Intestinal_Trajectories.png            # 肠道轨迹（3D）
└── Demo6_Intestinal_Velocity_Field.png          # 肠道速度场
```

## 技术细节

- **背侧分析器**: `DorsalIntercalationAnalyzer` 类
- **肠道分析器**: `IntestinalPrimordiumAnalyzer` 类  
- **统一入口**: `run_all_demos.py`
- **核心代码**: `dorsal_intercalation_analysis.py`

## 图表特征

### 正方形热图设计
- **Demo1A & Demo1B**: 左右分离的正方形热图（8×8英寸）
- **优势**: 更适合论文排版，左右对比清晰
- **字体**: 支持统一缩放，适应不同发表需求

### 字体缩放系统
```bash
# 论文用大字体
.venv/bin/python run_all_demos.py --font-scale 1.2

# 海报用超大字体  
.venv/bin/python run_all_demos.py --font-scale 1.8
```

所有demo数据都是理想化的模拟数据，用于展示期望的生物学模式和分析方法。