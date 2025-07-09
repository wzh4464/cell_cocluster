# Dorsal Intercalation Analysis

## Overview

This script analyzes dorsal intercalation cells from the `DATA/dorsal_intercalation.txt` file to generate visualizations matching the requirements specified in `/Volumes/Mac_Ext/link_cache/codes/cell_plot/README.md`.

## Features

The analysis generates four main visualizations:

### Panel A: Co-clustering Probability Heatmap

- **Output**: `dorsal_plots/PanelA_coclustering_heatmap.png`
- **Description**: Heatmap showing co-clustering probability for each dorsal cell over time (220-250 minutes)
- **X-axis**: Time (minutes post-fertilization)
- **Y-axis**: Dorsal cell IDs (DC01-DC20)
- **Color**: Co-clustering probability (darker = higher probability)

### Panel B: Cell Trajectory Visualization

- **Output**: `dorsal_plots/PanelB_cell_trajectories.png`
- **Description**: Movement trajectories of dorsal cells during intercalation
- **X-axis**: Anterior-Posterior Axis (μm)
- **Y-axis**: Left-Right Axis (μm)
- **Green circles**: Starting positions
- **Red squares**: Ending positions
- **Dashed line**: Midline reference

### Panel C: Morphological Irregularity Dynamics

- **Output**: `dorsal_plots/PanelC_morphological_irregularity.png`
- **Description**: Changes in cell shape irregularity over time
- **X-axis**: Time (minutes post-fertilization)
- **Y-axis**: Morphological irregularity index
- **Light blue lines**: Individual cell trajectories
- **Dark blue line**: Mean ± standard error
- **Yellow region**: Co-clustering active window

### Panel D: Velocity Field Analysis

- **Output**: `dorsal_plots/PanelD_velocity_field.png`
- **Description**: Box plots showing velocity distribution across time
- **X-axis**: Time (minutes post-fertilization)
- **Y-axis**: Midline crossing velocity (μm/min)
- **Box plots**: Statistical distribution of velocities at each time point

## Usage

```bash
# Make sure you're in the project directory
cd /Volumes/Mac_Ext/link_cache/codes/cell_cocluster

# Activate the virtual environment
source .venv/bin/activate

# Run the analysis
python dorsal_intercalation_analysis.py
```

## Output

The script will create a `dorsal_plots/` directory containing all four visualization panels. The console output will show:

1. Number of dorsal cells found in the input file
2. Number of cells successfully mapped to IDs
3. Progress updates for each visualization
4. Final summary of generated plots

## Data Sources

- **Cell Names**: `DATA/dorsal_intercalation.txt` - List of 20 dorsal intercalation cell names
- **Name Dictionary**: `DATA/name_dictionary.csv` - Maps cell names to numerical IDs
- **Segmentation Data**: `DATA/SegmentCellUnified/` - 3D segmentation files for each time point
- **Clustering Results**: `DATA/sub_triclusters/` - Co-clustering analysis results

## Current Status

- ✅ Successfully identified 12 out of 20 dorsal intercalation cells
- ✅ Generated all four required visualizations
- ✅ Handles 3D segmentation data correctly
- ✅ Robust error handling for missing data

## Notes

- Some dorsal cells from the input list could not be found in the name dictionary (8 out of 20)
- The script uses placeholder data for co-clustering probabilities - this should be replaced with actual clustering results
- 3D morphological analysis uses volume-based irregularity metrics instead of 2D perimeter
- All visualizations are saved as high-resolution PNG files (300 DPI)

## Dependencies

- numpy
- pandas
- matplotlib
- seaborn
- nibabel
- scikit-image
- scipy

## Future Improvements

1. **Real Co-clustering Data**: Replace placeholder probability calculations with actual clustering results
2. **Missing Cells**: Investigate why 8 dorsal cells are not found in the name dictionary
3. **Statistical Analysis**: Add statistical significance testing for velocity comparisons
4. **Interactive Plots**: Consider adding interactive visualizations using plotly
5. **Performance**: Optimize for large datasets if needed
