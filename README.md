# Cell Co-clustering Analysis Toolkit

一个用于分析和可视化3D显微镜数据中细胞共聚类模式的工具包。

## 项目概述

这个项目实现了一个完整的细胞共聚类分析流水线，专门用于分析3D显微镜数据中的细胞形态和运动特征。它结合了运动学特征（体积、表面积、质心、速度、加速度）和球谐波形状描述符来进行多层次的聚类分析。

## 核心功能

### 1. 特征提取

- **运动学特征**: 体积、表面积、质心坐标、速度向量、加速度向量
- **形状特征**: 15阶球谐波系数用于描述细胞表面形状
- **时序分析**: 支持时间序列数据的特征提取和轨迹分析

### 2. 聚类分析

- **双聚类**: 使用Spectral Co-clustering对细胞和时间点进行联合聚类
- **三聚类**: 使用CGC (Clustering Geodata Cubes)库对细胞、时间、特征三个维度进行聚类
- **并行处理**: 支持多进程并行计算以提高处理效率

### 3. 可视化

- **交互式可视化**: 基于Plotly的交互式聚类结果展示
- **时间序列动画**: 细胞形状演化的动画展示
- **聚类投影**: 将高维聚类结果投影到二维平面进行可视化

## 安装

### 环境要求

- Python >= 3.12
- 推荐使用uv进行包管理

### 安装步骤

```bash
# 克隆仓库
git clone https://github.com/yourusername/cell_cocluster.git
cd cell_cocluster

# 使用uv安装依赖
uv pip install -e .

# 或使用pip
pip install -e .
```

## 主要模块

### 数据预处理

- `aggregate_features.py`: 聚合运动学和球谐波特征
- `src/features/kinematic_features.py`: 运动学特征提取
- `src/data/reader.py`: NIfTI数据读取和处理

### 聚类分析

- `coclust_aggregated_features.py`: 主要的共聚类分析流水线
- `cocluster_timeline.py`: 时间线聚类可视化
- `sparse_tensor_analysis.py`: 稀疏张量分析

### 可视化

- `visualize_lineage.py`: 单个细胞谱系可视化
- `visualize_all_lineages.py`: 所有谱系的网格化可视化
- `visualize_cell_timeline.py`: 细胞时间线可视化
- `src/visualization/`: 可视化工具模块

## 使用方法

### 1. 特征聚合

```bash
# 从原始数据提取并聚合特征
python aggregate_features.py
```

### 2. 运行聚类分析

```bash
# 执行完整的共聚类分析
python coclust_aggregated_features.py
```

### 3. 生成可视化

```bash
# 生成单个谱系的动画
python visualize_lineage.py

# 生成所有谱系的网格动画
python visualize_all_lineages.py
```

### 4. 命令行工具

```bash
# 使用命令行工具提取运动学特征
kinematic-features --help
```

## 项目结构

```
cell_cocluster/
├── src/                          # 核心源代码
│   ├── data/                     # 数据读取和处理
│   ├── features/                 # 特征提取模块
│   ├── utils/                    # 工具函数
│   └── visualization/            # 可视化模块
├── tasks/                        # 任务脚本
├── tests/                        # 测试文件
├── configs/                      # 配置文件
├── DATA/                         # 数据目录
│   ├── geo_features/            # 运动学特征数据
│   ├── spharm/                  # 球谐波特征数据
│   └── name_dictionary.csv     # 细胞名称映射
├── docs/                         # 文档目录
├── aggregate_features.py         # 特征聚合脚本
├── coclust_aggregated_features.py # 主聚类分析脚本
├── cocluster_timeline.py        # 时间线聚类脚本
├── visualize_*.py               # 可视化脚本
└── pyproject.toml               # 项目配置
```

## 算法详情

### 特征空间

- **运动学维度**: 11个特征（体积、表面积、质心xyz、速度xyz、加速度xyz）
- **形状维度**: 225个球谐波系数（15阶）
- **时间维度**: 支持任意时间点序列

### 聚类策略

1. **第一层**: 对细胞-时间矩阵进行双聚类
2. **第二层**: 对每个双聚类子区域进行三聚类（细胞×时间×特征）
3. **后处理**: 聚类结果的可视化和统计分析

### SVD估计

使用奇异值分解(SVD)自动估计最优聚类数目。

## 配置文件

项目使用YAML配置文件(`configs/default.yaml`)管理参数：

```yaml
feature_extraction:
  nmf_components: 5
  cell_threshold: 0.5
  smoothing_sigma: 1.0

visualization:
  colormap: viridis
  figure_size: [10, 8]
  dpi: 300

paths:
  data_dir: "DATA"
  output_dir: "results"
  name_dictionary: "DATA/name_dictionary.csv"
```

## 依赖项

主要依赖包括：

- `numpy`, `pandas`, `scipy`: 数值计算
- `scikit-learn`: 机器学习算法
- `matplotlib`, `plotly`, `seaborn`: 可视化
- `nibabel`: NIfTI文件处理
- `pyshtools`: 球谐波计算
- `clustering-geodata-cubes`: 三聚类算法
- `dask`: 并行计算

## 许可证

MIT License

## 开发者

- **作者**: Zihan
- **邮箱**: <wzh4464@gmail.com>

## 版本历史

- **v0.1.0**: 初始版本，包含基本的特征提取和聚类功能
- 支持运动学特征和球谐波特征的提取
- 实现了双聚类和三聚类算法
- 提供了多种可视化工具
