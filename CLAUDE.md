# CLAUDE.md - 项目理解与配置

## Python 环境配置
- 使用 uv 管理 Python 环境
- Python 解释器路径: `.venv/bin/python`
- 运行命令示例: `.venv/bin/python script_name.py`

## 项目理解
这是一个细胞共聚类分析项目，专注于背侧嵌入(dorsal intercalation)细胞的时空分析。

### 核心功能
1. **共聚类概率分析** (`calculate_coclustering_probabilities`)
   - 计算背侧细胞在220-250分钟时间窗口内的共聚类概率
   - 生成热图可视化，显示细胞在不同时间点的聚类倾向

2. **细胞轨迹追踪** 
   - 提取细胞质心位置随时间的变化
   - 可视化细胞移动路径

3. **形态学不规则性分析**
   - 计算细胞形状不规则性指数
   - 分析发育过程中的形态变化

4. **速度场分析**
   - 计算细胞跨中线的速度
   - 分析背侧嵌入过程的动力学特征

### 数据结构
- 输入数据: 3D分割数据 (NIfTI格式)
- 时间范围: 220-250分钟 (受精后)
- 背侧细胞列表: `DATA/dorsal_intercalation.txt`
- 细胞名称映射: `DATA/name_dictionary.csv`
- 聚类结果: `DATA/sub_triclusters/all_subcluster_results.json`

### 理想的共聚类热图特征
好的结果应该显示:
1. **左右分群**: 背侧细胞分为左侧群和右侧群，各自有相似的聚类模式
2. **离散时间点**: 每分钟一个数据点，时间范围220-255分钟
3. **同时聚类**: 225-230分钟快速上升，230-254分钟高概率聚类
4. **左右分离**: 两幅独立热图，无标题设计
5. **高对比度**: 使用RdBu_r色彩方案，易于识别聚类活跃期

### Demo功能集成
已将demo代码集成到`DorsalIntercalationAnalyzer`类中:
- `create_demo_coclustering_data()`: 生成理想的聚类概率数据
- `plot_coclustering_heatmap(use_demo_data=True)`: 可选择使用demo数据
- 在`generate_all_plots()`中自动生成demo图表