import unittest
import os
import numpy as np
from pathlib import Path
import shutil
from src.features.zelin import get_SH_coefficient_of_embryo
import pandas as pd
from multiprocessing import Pool, cpu_count
from functools import partial
import nibabel as nib
import glob

def process_single_timepoint(source_file, target_cell, sample_N, lmax, name_dictionary_path, test_data_dir, test_output_dir):
    """Process a single timepoint file"""
    # Create a unique directory for this timepoint
    timepoint = source_file.stem.split("_")[-2]
    timepoint_dir = test_data_dir / f"tp_{timepoint}"
    timepoint_dir.mkdir(exist_ok=True)

    # Copy the timepoint file
    shutil.copy(source_file, timepoint_dir)

    # Create output directory for this cell
    cell_output_dir = test_output_dir / target_cell
    cell_output_dir.mkdir(exist_ok=True)

    # Calculate SH coefficients using non-parallel version
    get_SH_coefficient_of_embryo(
        embryos_path_root=str(timepoint_dir),
        saving_path_root=str(cell_output_dir),
        sample_N=sample_N,
        lmax=lmax,
        name_dictionary_path=name_dictionary_path,
        surface_average_num=3,
        target_cell=target_cell  # Add target cell parameter
    )

    # Check if the cell's data was saved for this timepoint
    cell_file = cell_output_dir / f"{target_cell}_{timepoint}_l{lmax+1}.npy"
    if cell_file.exists():
        data = np.load(cell_file)
        return timepoint, data
    return timepoint, None

class TestSHCoefficients(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        # Create test directories
        cls.test_data_dir = Path("test_data").resolve()
        cls.test_output_dir = Path("test_output").resolve()
        cls.test_data_dir.mkdir(exist_ok=True)
        cls.test_output_dir.mkdir(exist_ok=True)

        # Copy a single nii.gz file for testing (using the first available timepoint)
        source_file = Path("DATA/SegmentCellUnified/WT_Sample1LabelUnified/WT_Sample1LabelUnified_120_segCell.nii.gz").resolve()
        if source_file.exists():
            shutil.copy(source_file, cls.test_data_dir)
            print(f"Copied test file: {source_file}")
        else:
            print(f"Warning: Test file not found: {source_file}")

    def test_single_timepoint(self):  # sourcery skip: extract-method
        """Test SH coefficient calculation for a single timepoint"""
        # Copy a single nii.gz file for testing
        source_file = Path("DATA/SegmentCellUnified/WT_Sample1LabelUnified/WT_Sample1LabelUnified_120_segCell.nii.gz").resolve()
        if source_file.exists():
            shutil.copy(source_file, self.test_data_dir)
            print(f"Copied test file: {source_file}")
        else:
            print(f"Warning: Test file not found: {source_file}")

        # Parameters for testing
        sample_N = 30
        lmax = 14
        name_dictionary_path = Path("DATA/name_dictionary.csv").resolve()

        print(f"\nTest parameters:")
        print(f"Sample N: {sample_N}")
        print(f"Lmax: {lmax}")
        print(f"Name dictionary path: {name_dictionary_path}")
        print(f"Test data dir: {self.test_data_dir}")
        print(f"Test output dir: {self.test_output_dir}")

        # Check if input files exist
        nii_files = list(self.test_data_dir.glob("*.nii.gz"))
        print(f"\nFound {len(nii_files)} NIfTI files:")
        for file in nii_files:
            print(f"  - {file}")

        if not nii_files:
            self.fail("No NIfTI files found in test data directory")

        if not name_dictionary_path.exists():
            self.fail(f"Name dictionary file not found: {name_dictionary_path}")

        # Run the function
        print("\nRunning get_SH_coefficient_of_embryo...")
        try:
            get_SH_coefficient_of_embryo(
                embryos_path_root=str(self.test_data_dir),
                saving_path_root=str(self.test_output_dir),
                sample_N=sample_N,
                lmax=lmax,
                name_dictionary_path=str(name_dictionary_path),
                surface_average_num=3
            )
        except Exception as e:
            print(f"Error running get_SH_coefficient_of_embryo: {str(e)}")
            raise

        # Check if output files were created
        output_files = list(self.test_output_dir.glob("**/*.npy"))
        print(f"\nFound {len(output_files)} output files:")
        for file in output_files:
            print(f"  - {file}")

        self.assertGreater(len(output_files), 0, "No output files were created")

        # Check the content of the first output file
        if output_files:
            data = np.load(output_files[0])
            self.assertEqual(data.ndim, 1, "SH coefficients should be a 1D array")
            expected_size = (lmax + 1) ** 2
            self.assertEqual(data.size, expected_size, 
                           f"SH coefficients array should have size {(lmax + 1) ** 2}")

            # Print some information about the test
            print(f"\nTest completed successfully:")
            print(f"Output file shape: {data.shape}")
            print(f"Expected size: {expected_size}")

    def test_cell_across_timepoints_parallel(self):
        """Test SH coefficient calculation for a specific cell across all timepoints using parallel processing"""
        # Parameters
        target_cell = "AB"  # Changed from MSaaaap to AB
        sample_N = 30
        lmax = 14
        name_dictionary_path = "DATA/name_dictionary.csv"

        # Get all timepoint files and take only first 10
        source_dir = Path("DATA/SegmentCellUnified/WT_Sample1LabelUnified")
        timepoint_files = sorted(source_dir.glob("*_segCell.nii.gz"))[:10]

        print(f"\nProcessing cell {target_cell} across first {len(timepoint_files)} timepoints in parallel")

        # Create output directory for this cell
        cell_output_dir = self.test_output_dir / target_cell
        cell_output_dir.mkdir(exist_ok=True)

        # Prepare the processing function with fixed parameters
        process_func = partial(
            process_single_timepoint,
            target_cell=target_cell,
            sample_N=sample_N,
            lmax=lmax,
            name_dictionary_path=name_dictionary_path,
            test_data_dir=self.test_data_dir,
            test_output_dir=self.test_output_dir,
        )

        # Use multiprocessing to process timepoints in parallel
        num_processes = min(cpu_count(), len(timepoint_files))
        print(f"Using {num_processes} processes for parallel processing")

        with Pool(num_processes) as pool:
            results = pool.map(process_func, timepoint_files)

        # Collect results
        sh_coefficients = []
        timepoints = []

        for timepoint, data in results:
            if data is not None:
                sh_coefficients.append(data)
                timepoints.append(timepoint)
                print(f"Found cell in timepoint {timepoint}, SH coefficients shape: {data.shape}")
            else:
                print(f"Cell not found in timepoint {timepoint}")
                print(f"Found {results}")

        # Convert results to DataFrame
        if sh_coefficients:
            sh_array = np.stack(sh_coefficients)
            column_names = [f"SH_{i}" for i in range(sh_array.shape[1])]
            df = pd.DataFrame(sh_array, index=timepoints, columns=column_names)

            # Save results
            results_file = cell_output_dir / f"{target_cell}_SH_coefficients.csv"
            df.to_csv(results_file)

            print(f"\nResults saved to {results_file}")
            print(f"Shape of results: {df.shape}")
            print("\nFirst few timepoints:")
            print(df.head())

            # Verify results
            self.assertGreater(len(sh_coefficients), 0, "No SH coefficients were calculated")
            self.assertEqual(sh_array.shape[1], (lmax + 1) ** 2, 
                           "Incorrect number of SH coefficients")

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        # Remove test directories
        if cls.test_data_dir.exists():
            shutil.rmtree(cls.test_data_dir)
        if cls.test_output_dir.exists():
            shutil.rmtree(cls.test_output_dir)

if __name__ == '__main__':
    unittest.main() 
