"""
Task for extracting kinematic features from cell data.
"""

import click
from pathlib import Path
from src.features.kinematic_features import extract_cell_features

@click.command()
@click.argument('data_dir', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
@click.option('--target-cell', type=str, help='Target cell ID to process')
@click.option('--timepoints', type=str, help='Comma-separated list of timepoints to process')
def main(data_dir: str, output_dir: str, target_cell: str = None, timepoints: str = None):
    """
    Extract kinematic features from cell data.

    Args:
        data_dir: Directory containing NIfTI files
        output_dir: Directory to save feature files
        target_cell: Optional target cell ID to process
        timepoints: Optional comma-separated list of timepoints to process
    """
    # Convert timepoints string to list if provided
    timepoints_list = timepoints.split(',') if timepoints else None

    print("Extracting cell features...")
    stats = extract_cell_features(
        data_dir=data_dir,
        output_dir=output_dir,
        target_cell=target_cell,
        timepoints=timepoints_list
    )

    print("\nExtraction Statistics:")
    print(f"Total cells processed: {stats['total_cells']}")
    print(f"Timepoints processed: {stats['processed_timepoints']}")
    print(f"Unique cells processed: {len(stats['processed_cells'])}")

if __name__ == "__main__":
    main()
