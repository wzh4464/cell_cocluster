import unittest
import os
import numpy as np
from pathlib import Path
import shutil
import nibabel as nib
from src.features.kinematic_features import extract_cell_features, calculate_cell_features
from src.utils.file_utils import get_timepoint_from_filename, get_cell_id_from_label
import pandas as pd

class TestExtractor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        # Create test directories
        cls.test_data_dir = Path("DATA/SegmentCellUnified/WT_Sample1LabelUnified")
        cls.test_output_dir = Path("DATA/geo_features")
        cls.test_output_dir.mkdir(exist_ok=True)

        # Verify test data exists
        if not cls.test_data_dir.exists():
            raise FileNotFoundError(f"Test data directory not found: {cls.test_data_dir}")
        
        # Verify NIfTI files exist and get first 10 timepoints
        cls.nifti_files = sorted(list(cls.test_data_dir.glob("*_segCell.nii.gz")))[:10]
        if not cls.nifti_files:
            raise FileNotFoundError(f"No NIfTI files found in {cls.test_data_dir}")
        
        print(f"Using {len(cls.nifti_files)} timepoints for testing")

    def test_calculate_cell_features(self):
        """Test calculation of cell features"""
        # Create a simple test volume
        volume = np.zeros((10, 10, 10))
        volume[2:8, 2:8, 2:8] = 1  # Create a cube
        
        # Calculate features
        features = calculate_cell_features(volume, 1)
        
        # Check feature types
        self.assertIn('volume', features)
        self.assertIn('surface_area', features)
        self.assertIn('centroid_x', features)
        self.assertIn('centroid_y', features)
        self.assertIn('centroid_z', features)
        
        # Check feature values
        self.assertEqual(features['volume'], 216)  # 6x6x6 cube
        self.assertGreater(features['surface_area'], 0)
        self.assertEqual(features['centroid_x'], 4.5)
        self.assertEqual(features['centroid_y'], 4.5)
        self.assertEqual(features['centroid_z'], 4.5)

    def test_extract_cell_features(self):
        """Test feature extraction for all cells"""
        # Run feature extraction
        stats = extract_cell_features(
            data_dir=str(self.test_data_dir),
            output_dir=str(self.test_output_dir),
            timepoints=[get_timepoint_from_filename(f) for f in self.nifti_files]  # Only use first 10 timepoints
        )
        
        # Check statistics
        self.assertGreater(stats['total_cells'], 0, "No cells were processed")
        self.assertGreater(stats['processed_timepoints'], 0, "No timepoints were processed")
        self.assertGreater(len(stats['processed_cells']), 0, "No unique cells were processed")
        
        # Check output files
        cell_dirs = list(self.test_output_dir.glob("cell_*"))
        self.assertGreater(len(cell_dirs), 0, "No cell directories were created")
        
        for cell_dir in cell_dirs:
            # Check feature files
            feature_files = list(cell_dir.glob("*_features.npy"))
            self.assertGreater(len(feature_files), 0, f"No feature files found in {cell_dir}")
            
            # Check metadata files
            metadata_files = list(cell_dir.glob("*_metadata.npy"))
            self.assertGreater(len(metadata_files), 0, f"No metadata files found in {cell_dir}")
            
            # Load and check a feature file
            features = np.load(feature_files[0], allow_pickle=True).item()
            self.assertIn('volume', features, "Volume feature missing")
            self.assertIn('surface_area', features, "Surface area feature missing")
            self.assertIn('velocity_x', features, "Velocity feature missing")
            self.assertIn('acceleration_x', features, "Acceleration feature missing")

    def test_extract_single_cell(self):
        """Test feature extraction for a single cell"""
        # Get a list of available cells from the first timepoint
        first_file = self.nifti_files[0]
        nii_img = nib.load(first_file)
        volume = nii_img.get_fdata()
        available_cells = np.unique(volume)[1:]  # Skip background
        print(f"Available cells in first timepoint: {available_cells}")
        
        if len(available_cells) == 0:
            raise ValueError("No cells found in the first timepoint")
            
        target_cell = get_cell_id_from_label(int(available_cells[0]))  # Use the first available cell
        print(f"Target cell: {target_cell}")
        
        # Run feature extraction
        stats = extract_cell_features(
            data_dir=str(self.test_data_dir),
            output_dir=str(self.test_output_dir),
            target_cell=target_cell,
            timepoints=[get_timepoint_from_filename(f) for f in self.nifti_files]  # Only use first 10 timepoints
        )
        
        print(f"Extraction stats: {stats}")
        
        # Check statistics
        self.assertEqual(stats['total_cells'], 1, "Should only process one cell")
        self.assertGreater(stats['processed_timepoints'], 0, "No timepoints were processed")
        self.assertEqual(len(stats['processed_cells']), 1, "Should only process one unique cell")
        
        # Check output files
        cell_dir = self.test_output_dir / target_cell
        self.assertTrue(cell_dir.exists(), f"Cell directory {cell_dir} not found")
        
        # Check feature files
        feature_files = list(cell_dir.glob("*_features.npy"))
        self.assertGreater(len(feature_files), 0, f"No feature files found in {cell_dir}")
        
        # Check metadata files
        metadata_files = list(cell_dir.glob("*_metadata.npy"))
        self.assertGreater(len(metadata_files), 0, f"No metadata files found in {cell_dir}")

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        # We don't remove the output directory as it contains the actual features
        pass

if __name__ == '__main__':
    unittest.main() 
