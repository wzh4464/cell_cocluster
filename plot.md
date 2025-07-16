好的，下面我将详细说明 dorsal_intercalation_analysis.py 脚本中每张图的数据来源和处理流程。

---

## 数据来源

1. **dorsal_intercalation.txt**  
   路径：DATA/dorsal_intercalation.txt  
   内容：包含所有背侧插入（dorsal intercalation）细胞的名称，每行一个细胞名。

2. **name_dictionary.csv**  
   路径：DATA/name_dictionary.csv  
   内容：细胞名称与细胞ID的映射表。脚本用它将 dorsal_intercalation.txt 里的细胞名转换为对应的细胞ID。

3. **分割数据（细胞分割nii.gz）**  
   路径：DATA/SegmentCellUnified/WT_Sample1LabelUnified/WT_Sample1LabelUnified_{time}_segCell.nii.gz  
   内容：每个时间点的三维细胞分割数据，文件名中的 {time} 是三位数的时间点编号（如220、221...250）。  
   用于提取每个细胞在每个时间点的空间位置和形态。

4. **聚类结果（可选）**  
   路径：DATA/sub_triclusters/all_subcluster_results.json  
   内容：细胞三聚类分析的结果（如果存在），用于共聚类概率的计算。

---

## 每张图的数据来源和处理流程

### 1. 共聚类概率热图（Co-clustering Probability Heatmap）

- **数据来源**：  
  - dorsal_intercalation.txt（细胞名）
  - name_dictionary.csv（细胞ID）
  - sub_triclusters/all_subcluster_results.json（聚类结果，若无则用随机数占位）

- **处理流程**：  
  1. 读取 dorsal_intercalation.txt，获得所有目标细胞名。
  2. 用 name_dictionary.csv 映射为细胞ID。
  3. 读取聚类结果（如果有），否则用随机数填充。
  4. 构建一个“细胞数 × 时间点数”的概率矩阵，表示每个细胞在每个时间点的共聚类概率。
  5. 用 seaborn 画热图，x轴为时间点，y轴为细胞，颜色表示概率。

- **脚本相关函数**：  
  - load_dorsal_cells
  - load_name_dictionary
  - find_dorsal_cell_ids
  - calculate_coclustering_probabilities
  - plot_coclustering_heatmap

---

### 2. 细胞轨迹图（Cell Trajectory Visualization）

- **数据来源**：  
  - dorsal_intercalation.txt（细胞名）
  - name_dictionary.csv（细胞ID）
  - SegmentCellUnified/WT_Sample1LabelUnified/WT_Sample1LabelUnified_{time}_segCell.nii.gz（分割数据）

- **处理流程**：  
  1. 读取 dorsal_intercalation.txt 和 name_dictionary.csv，获得目标细胞ID。
  2. 对每个时间点，读取对应的分割nii.gz文件。
  3. 对每个细胞，提取其在每个时间点的三维质心坐标（ndimage.center_of_mass）。
  4. 汇总每个细胞的轨迹（时间+空间坐标）。
  5. 画出每个细胞的二维轨迹（x: 前后轴，y: 左右轴），起点绿色圆点，终点红色方块。

- **脚本相关函数**：  
  - extract_cell_centroids
  - calculate_trajectories
  - plot_cell_trajectories

---

### 3. 形态不规则性动态（Morphological Irregularity Dynamics）

- **数据来源**：  
  - dorsal_intercalation.txt（细胞名）
  - name_dictionary.csv（细胞ID）
  - SegmentCellUnified/WT_Sample1LabelUnified/WT_Sample1LabelUnified_{time}_segCell.nii.gz（分割数据）

- **处理流程**：  
  1. 读取 dorsal_intercalation.txt 和 name_dictionary.csv，获得目标细胞ID。
  2. 对每个时间点，读取分割nii.gz文件。
  3. 对每个细胞，提取其分割mask，计算体积（体素数）和表面积proxy（用面积的平方根近似）。
  4. 计算不规则性指标：surface_proxy / (volume的立方根)。
  5. 汇总所有细胞的指标，画出每个细胞的曲线，并计算均值±标准误，画出主趋势线和误差带。

- **脚本相关函数**：  
  - calculate_morphological_irregularity
  - plot_morphological_irregularity

---

### 4. 速度场分析（Velocity Field Analysis）

- **数据来源**：  
  - dorsal_intercalation.txt（细胞名）
  - name_dictionary.csv（细胞ID）
  - SegmentCellUnified/WT_Sample1LabelUnified/WT_Sample1LabelUnified_{time}_segCell.nii.gz（分割数据）

- **处理流程**：  
  1. 读取 dorsal_intercalation.txt 和 name_dictionary.csv，获得目标细胞ID。
  2. 提取每个细胞在每个时间点的质心坐标。
  3. 计算每个细胞在x轴（前后轴）方向的速度（相邻时间点的x坐标差/时间差）。
  4. 按时间分箱（每2分钟），统计所有细胞在该时间段的速度，画成箱线图。

- **脚本相关函数**：  
  - calculate_midline_velocity
  - plot_velocity_field

---

## 总结

- 所有分析的核心原始数据是 dorsal_intercalation.txt（目标细胞名）、name_dictionary.csv（ID映射）、分割nii.gz（细胞空间信息），部分分析还用到聚类json（共聚类概率）。
- 每张图都基于这些数据，经过不同的处理和统计，展示细胞的空间轨迹、形态变化、聚类关系和运动速度等生物学特征。

如需某一部分的详细代码解读或数据结构说明，请告知！