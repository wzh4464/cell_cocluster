# Cell Co-clustering Analysis Toolkit

A toolkit for analyzing and visualizing cell co-clustering patterns in 3D microscopy data.

## Features

- 3D cell feature extraction
- Non-negative matrix factorization for clustering
- Interactive visualization using Dash
- Support for various microscopy data formats

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/cell_cocluster.git
cd cell_cocluster

# Install dependencies
pip install -e .
```

## Usage

```python
from cell_cocluster.features.extractor import construct_feature_tensor
from cell_cocluster.visualization.plotter import visualize_clusters

# Extract features
features = construct_feature_tensor("path/to/data")

# Visualize results
visualize_clusters(features)
```

## Project Structure

```
cell_cocluster/
├── src/                    # Source code
├── tests/                 # Test files
├── DATA/                  # Data directory
├── configs/              # Configuration files
└── notebooks/           # Example notebooks
```

## License

MIT License 