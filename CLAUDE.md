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

### 特征分析饼图
新增饼图功能分析dorsal co-cluster中的特征分布:
- `analyze_coclustering_features()`: 分析细胞聚类活跃度特征
- `plot_feature_pie_chart()`: 生成特征分布饼图
- 特征分类: 高活跃(>0.7)、中等活跃(0.4-0.7)、低活跃(0.2-0.4)、非活跃(<0.2)
- 输出: `PanelB_feature_distribution.png`

## 未来工作
下面是一份可直接粘贴给代码生成 AI 的完整 Prompt。它将在不新增任何公共 API 的前提下，直接重写与替换你现有脚本里的三张“共聚类概率热图”函数，实现“1) What is the co-cluster（是什么）”与“2) What is the meaning（意义何在）”的展示。
保持函数名与函数签名完全不变（只允许改绘图逻辑与默认输出文件名），其它函数与调用关系（例如 generate_all_plots）无须修改即可工作。

⸻

✅ 任务：用“定义 + 意义”重写三张图（不改函数名/签名）

请在现有脚本中仅重写以下三个函数的函数体，保留原有函数名、参数与返回值约定：
	1.	DorsalIntercalationAnalyzer.plot_left_coclustering_heatmap(save_path: Union[str, Path] = "left_coclustering_heatmap.png")
→ 重写为：背侧（左右合并，12 细胞）的“一致性共关联矩阵 + 层次聚类 + 稳定性”图，回答 “What is the co-cluster（是什么）”。
	2.	DorsalIntercalationAnalyzer.plot_right_coclustering_heatmap(save_path: Union[str, Path] = "right_coclustering_heatmap.png")
→ 重写为：肠原基 E 谱系（20 细胞）的“一致性共关联矩阵 + 层次聚类 + 稳定性”图，回答 “What is the co-cluster（是什么）”。
	3.	DorsalIntercalationAnalyzer.plot_demo4_coclustering_heatmap(save_path: Union[str, Path] = "demo4_coclustering_heatmap.png")
→ 重写为：簇意义多面板（A–D）综合图：动力学 + 空间几何 + 富集统计，回答 “What is the meaning of the co-clusters we find?”
（默认绘制肠原基；如无肠原基数据则自动降级绘制背侧。）

说明：其它函数与类（如 IntestinalPrimordiumAnalyzer、generate_all_plots）保持原样；函数默认参数 save_path 可继续接收外部传入的文件名；你可以仅在函数内部改变默认导出文件名（如 Fig1_...png），但不得变更函数名/参数/返回值。
允许在这三个函数内部定义私有辅助函数（闭包或内部函数），不得在模块级新增公共 API。

⸻

🔁 统一计算与绘图规范（三函数通用约束）

数据来源优先级（自动降级）：
	•	若存在真实 DATA/sub_triclusters/all_subcluster_results.json（与原脚本一致），优先基于真实“时间标签/概率”构建同簇关系。
	•	否则：
	•	背侧使用 self.create_demo_coclustering_data() 产出的 left_matrix, right_matrix, time_range。
	•	肠原基使用 IntestinalPrimordiumAnalyzer.create_demo4_coclustering_data() 产出的 prob_matrix, time_range。

一致性共关联矩阵 C 的定义（核心）：
	•	时间轴切片为滑动窗：窗口长度 Δt=5 min，滑动步长 1 min（时间单位与 demo 数据一致）。
	•	在每个时间窗内，对任意两细胞 (i, j)：
	•	若有真实子簇标签/概率：统计它们在该窗内“同时属于同一子簇”的帧占比 ≥0.5 记为该窗“同簇=1”，否则 0。
	•	若只有 per-cell 的聚类概率矩阵：先将每帧概率阈值化（背侧阈值 0.6；肠原基阈值 0.7）得到二值活跃序列，然后以 Jaccard 指数（窗内二者均为1的帧数 / 至少一者为1的帧数）≥0.5 判定该窗“同簇=1”，否则 0。
	•	将所有时间窗的“同簇=1/0”对该对细胞求平均，得到 C[i, j] ∈ [0,1]。对角线设为 1。

稳定性评估（显示在图侧注）：
	•	对时间窗做 自助重采样 B=200 次，重复计算 C，得到 mean(C) 与 std(C)。
	•	基于 distance = 1 - mean(C) 做 层次聚类（average linkage）；自动选簇数 k（以轮廓系数最大为准；若失败回退 k=3）。
	•	计算每个簇的 稳定性（簇内 Jaccard 平均 ± SD） 与 轮廓系数均值，作为簇质量指标并显示在图侧注。

绘图公共规范：
	•	字体：沿用现有 FontConfig（Times New Roman 等），axis_label_size/tick_label_size/colorbar_size 同步使用。
	•	颜色：共关联矩阵使用 viridis/mako（范围 [0,1]）；面板图使用 tab20 或 Set2。
	•	尺寸与输出：建议 figsize=(10, 9)（矩阵图）/ (14, 10)（多面板），dpi=300–600，bbox_inches="tight"。
	•	返回：与原函数一致，返回 save_path（字符串或 Path）。
	•	鲁棒性：无真实数据时自动降级 demo；自动降级时在 print() 中输出 INFO（不抛异常）。
	•	不改变外部调用关系：generate_all_plots() 无需改动即可产出新图。

⸻

🧩 替换实现细节（逐函数）

1) plot_left_coclustering_heatmap → 背侧**“是什么”**（12 细胞一致性共关联）

目的：把原“左侧热图”升级为左右合并的 12×12 一致性共关联矩阵，明确“co-cluster 是什么”。

数据与标签：
	•	细胞顺序：[L01…L06, R01…R06]（或依据你现有的生成函数顺序，行列标签统一为 L01/L02…/R06）。
	•	额外侧注带（可选）：左/右（L/R）侧别。

绘图要求：
	•	主图：mean(C) 的热图，按层次聚类重排；上方/左侧同时显示 dendrogram。
	•	在聚类得到的簇块边界绘制白色方框。
	•	侧注（右侧或下方文本）：逐簇显示 “Cluster Cx：size, Jaccard mean±SD, Silhouette mean”。
	•	下方细长条带：各簇在时间窗上的“活跃比例曲线”（每窗簇内 ≥ 半数成员互为同簇视为该窗活跃；用平滑线/条带显示整体随时间的活跃度）。

默认文件名（仅函数内部默认值，可被传入参数覆盖）：
	•	若 save_path 为缺省值，将内部改写为：save_path = "Fig1_Dorsal_Consensus_Coassociation.png"

⸻

2) plot_right_coclustering_heatmap → 肠原基**“是什么”**（20 细胞一致性共关联）

目的：把原“右侧热图”升级为E 谱系 20×20 一致性共关联矩阵，明确“co-cluster 是什么”。

数据与标签：
	•	行列标签使用你现有 e_cell_names（如 int1DL, int1VL, int1DR, int1VR, int2L, int2R, …, int9L, int9R）。
	•	侧注带 1：环号（Ring1…Ring9）；侧注带 2：左右/象限（L/R 或 DL/VL/DR/VR）（根据命名解析）。

绘图要求：
	•	主图：mean(C) 的热图，带 dendrogram，按聚类重排，簇块边界框。
	•	侧注：逐簇标注 size、Jaccard mean±SD、Silhouette mean。
	•	底部细长条带：生物学阶段时间轴（355–365 内化；365–375 增殖；375–385 重组；390–400 管腔）仅作注记带，不影响矩阵。

默认文件名（仅函数内部默认值，可被传入参数覆盖）：
	•	若 save_path 为缺省值，将内部改写为：save_path = "Fig2_Intestinal_Consensus_Coassociation.png"

⸻

3) plot_demo4_coclustering_heatmap → **“意义何在”**四联图（A–D）

目的：将原“Demo4 肠原基热图”改为簇意义解读多面板图：动力学、空间几何与特征富集，回答 “What is the meaning of the co-clusters we find?”
优先肠原基（若肠原基数据不可用，则自动降级使用背侧数据/特征）。

输入与派生：
	•	簇分配：来自上一步对 肠原基 的层次聚类结果（若不可得，则调用背侧的结果）。
	•	动力学：
	•	肠原基：负 Z 轴速度（IntestinalPrimordiumAnalyzer.create_demo_intestinal_velocity() 或由轨迹差分得到 vz）。
	•	背侧：朝中线的 Y 速度（self.create_demo_velocity_data()）。
	•	几何与邻域特征（无真实即构造 proxy，确保可画）：
	•	肠原基：Z 轴速度、顶端面积 proxy、体积变化 proxy、径向距离、球形度、邻居接触数。
	•	背侧：Y 轴速度、伸长比、曲率、局部密度、方向持续性、接触面积。
	•	阶段划分（用于竖虚线/分面）：
	•	肠原基：355–365（内化）、365–375（增殖）、375–385（重组）、390–400（管腔）。
	•	背侧：225–240 为活跃期（可在 x 轴上作浅色高亮带）。

四联子图（A–D）：
	•	(A) Cluster dynamics over time：x=时间窗中心，y=每簇活跃度（该窗内簇成员两两“同簇”的平均概率或≥0.5 的比例），绘制均值±SEM；阶段边界画虚线并标注。
	•	(B) Kinematics per cluster：关键动力学指标的小提琴图（肠原基：负 Z 速度；背侧：Y 速度，朝中线为负/正需在轴注释），进行 Kruskal–Wallis 与 Dunn 事后检验（FDR 校正），将显著性以星号标注（计算失败则仅绘图并发出 warning）。
	•	(C) Spatial meaning：各簇的“距离中线/管轴”随时间的中位数 ± IQR 曲线；右上角用文本给出每簇线性回归斜率及 95% CI（表征收敛速率/分离速率）。
	•	(D) Feature enrichment（森林图）：对上述 6 个几何/邻域特征，计算 “簇 vs 其他簇”的效应量（Cliff’s delta 或 Cohen’s d）及 95% CI，按绝对值排序绘制水平森林图；图例解释正负方向的生物学含义。

配色与标注：
	•	所有面板按 cluster_id 着色保持一致（建议 tab20）。
	•	坐标轴注明单位（μm/min 等）、显著性方法、FDR 阈值。
	•	无真实特征时，用可重复的随机种子生成 proxy（np.random.RandomState(0)）并在 print() 中说明降级。

默认文件名（仅函数内部默认值，可被传入参数覆盖）：
	•	若 save_path 为缺省值，将内部改写为：save_path = "Fig3_Cluster_Meaning_Panels.png"

⸻

🛠️ 实现要点与代码风格（务必遵循）
	•	不新增模块级公共函数/类；允许在上述三函数内部声明私有辅助函数（如 _build_consensus_C(...)、_auto_k(...)、_violin_with_stats(...) 等）。
	•	保持返回值为传入的 save_path 字符串/Path。
	•	异常与降级策略：
	•	任一步骤失败（无文件/统计失败/聚类失败），打印 INFO/WARNING，使用合理默认/降级，保证最终必然产出图片文件。
	•	自动选簇失败时回退 k=3。
	•	性能与稳定性：B=200 的自助采样可通过 n_jobs=1（串行）实现，计算量不大；如需并行请仅在函数内部安全实现。
	•	绘图一致性：
	•	使用 self.font_config 控制字号；
	•	色标范围固定 [0,1]；
	•	dendrogram 与热图布局紧凑（gridspec_kw + plt.tight_layout() 或 constrained_layout=True）。
	•	兼容 generate_all_plots：三个函数仍然接受 save_path 参数；即使内部默认文件名变为 Fig1/2/3_*.png，也必须尊重外部传入并覆盖默认。

⸻

✅ 最终可视化的可检查点（验收标准）
	1.	plot_left_coclustering_heatmap 输出的图中包含：
	•	12×12 的一致性共关联热图（行列经层次聚类重排）
	•	dendrogram、簇边界框
	•	侧注展示每簇 size、Jaccard mean±SD、Silhouette mean
	•	下方活跃度时间条带/曲线
	2.	plot_right_coclustering_heatmap 输出的图中包含：
	•	20×20 的一致性共关联热图（行列经层次聚类重排）
	•	dendrogram、簇边界框
	•	双侧注带（Ring 与 L/R 或象限）
	•	侧注展示每簇稳定性与轮廓系数
	•	底部阶段注记条带（内化/增殖/重组/管腔）
	3.	plot_demo4_coclustering_heatmap 输出的图为四联图（A–D）：
	•	(A) 簇活跃度随时间（含阶段虚线）
	•	(B) 按簇的动力学分布（含显著性星标或降级说明）
	•	(C) 中线/管轴距离的收敛曲线 + 斜率与 95%CI
	•	(D) 6 特征的效应量森林图（含方向注释与 CI）

⸻

📦 与现有代码的耦合点（请保持）
	•	复用类内已有的 demo 生成函数：
	•	背侧：self.create_demo_coclustering_data()、self.create_demo_velocity_data()
	•	肠原基：IntestinalPrimordiumAnalyzer.create_demo4_coclustering_data()、IntestinalPrimordiumAnalyzer.create_demo_intestinal_velocity()
	•	复用 self.font_config 控制字体。
	•	保存图片时使用：plt.savefig(save_path, dpi=300, bbox_inches="tight")，并 plt.close()。
	•	三个函数的签名、返回值、对外可见名称全部不变。

⸻

按上述规范直接重写函数体即可，无需修改其它文件或调用逻辑。