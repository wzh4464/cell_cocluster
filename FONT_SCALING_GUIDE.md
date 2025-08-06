# 字体缩放使用指南

## 概述

新的统一字体配置系统支持通过命令行参数调整所有demo图表的字体大小，适应不同的使用场景（论文、演示、海报等）。

## 使用方法

### 基本命令

```bash
# 默认字体大小 (缩放因子 1.0)
.venv/bin/python run_all_demos.py

# 论文用大字体 (推荐 1.2-1.5倍)
.venv/bin/python run_all_demos.py --font-scale 1.2

# 海报用超大字体 (推荐 1.5-2.0倍)
.venv/bin/python run_all_demos.py --font-scale 1.8

# 小尺寸图表 (推荐 0.7-0.9倍)
.venv/bin/python run_all_demos.py --font-scale 0.8
```

### 自定义输出目录

```bash
# 指定输出目录
.venv/bin/python run_all_demos.py --font-scale 1.5 --output-dir poster_plots

# 生成多个版本
.venv/bin/python run_all_demos.py --font-scale 1.2 --output-dir paper_plots
.venv/bin/python run_all_demos.py --font-scale 1.8 --output-dir poster_plots
.venv/bin/python run_all_demos.py --font-scale 0.8 --output-dir small_plots
```

## 字体大小对照表

| 缩放因子 | 轴标签 | 刻度标签 | 图例 | 色标 | 适用场景 |
|---------|--------|----------|------|------|----------|
| 0.8     | 12pt   | 11pt     | 9pt  | 11pt | 小图、内嵌图 |
| 1.0     | 16pt   | 14pt     | 12pt | 14pt | 默认大小 |
| 1.2     | 19pt   | 16pt     | 14pt | 16pt | 论文投稿 |
| 1.5     | 24pt   | 21pt     | 18pt | 21pt | 期刊论文 |
| 1.8     | 28pt   | 25pt     | 21pt | 25pt | 会议海报 |
| 2.0     | 32pt   | 28pt     | 24pt | 28pt | 大型展示 |

## 推荐设置

### 📖 期刊论文
```bash
.venv/bin/python run_all_demos.py --font-scale 1.2 --output-dir journal_figures
```
- 字体清晰易读
- 符合大部分期刊要求
- 轴标签19pt，刻度16pt

### 📊 会议海报
```bash
.venv/bin/python run_all_demos.py --font-scale 1.8 --output-dir poster_figures
```
- 远距离可读
- 突出重点信息
- 轴标签28pt，刻度25pt

### 📝 投稿预览
```bash
.venv/bin/python run_all_demos.py --font-scale 1.0 --output-dir submission_preview
```
- 标准大小
- 快速预览效果
- 轴标签16pt，刻度14pt

### 🖥️ 演示文稿
```bash
.venv/bin/python run_all_demos.py --font-scale 1.5 --output-dir presentation_slides
```
- 投影清晰
- 观众易读
- 轴标签24pt，刻度21pt

## 字体配置详情

系统使用 `FontConfig` 类统一管理所有字体设置：

- **轴标签**: `16 × scale_factor` pt，粗体
- **刻度标签**: `14 × scale_factor` pt，常规
- **图例**: `12 × scale_factor` pt，常规
- **色标**: `14 × scale_factor` pt，粗体

所有字体大小会自动取整到最近的整数。

## 帮助信息

```bash
.venv/bin/python run_all_demos.py --help
```

输出所有可用选项和使用说明。