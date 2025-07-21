"""
Tests for src.features.kinematic_features module.
"""

import numpy as np
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, call

from src.features.kinematic_features import (
    calculate_cell_features,
    calculate_velocity, 
    calculate_acceleration,
    process_single_timepoint,
    extract_cell_features
)


class TestCalculateCellFeatures:
    """Test calculate_cell_features function."""
    
    def test_calculate_cell_features_basic(self):
        """Test basic cell feature calculation."""
        # Create a simple test volume with one cell
        volume = np.zeros((10, 10, 10), dtype=int)
        volume[3:7, 3:7, 3:7] = 1  # 4x4x4 cube with label 1
        
        features = calculate_cell_features(volume, 1)
        
        # Check that all expected features are present
        expected_keys = ['volume', 'surface_area', 'centroid_x', 'centroid_y', 'centroid_z']
        assert all(key in features for key in expected_keys)
        
        # Volume should be 64 voxels (4x4x4)
        assert features['volume'] == 64
        
        # Centroid should be at the center of the cube
        assert abs(features['centroid_x'] - 4.5) < 0.1
        assert abs(features['centroid_y'] - 4.5) < 0.1
        assert abs(features['centroid_z'] - 4.5) < 0.1
        
        # Surface area should be positive
        assert features['surface_area'] > 0
    
    def test_calculate_cell_features_single_voxel(self):
        """Test with single voxel cell."""
        volume = np.zeros((5, 5, 5), dtype=int)
        volume[2, 2, 2] = 1
        
        features = calculate_cell_features(volume, 1)
        
        assert features['volume'] == 1
        assert features['centroid_x'] == 2.0
        assert features['centroid_y'] == 2.0
        assert features['centroid_z'] == 2.0
        assert features['surface_area'] >= 0
    
    def test_calculate_cell_features_no_cell(self):
        """Test with non-existent cell label."""
        volume = np.zeros((5, 5, 5), dtype=int)
        volume[2, 2, 2] = 1
        
        features = calculate_cell_features(volume, 999)  # Non-existent label
        
        # Should return zero volume and NaN centroids
        assert features['volume'] == 0
        assert features['surface_area'] == 0
        assert np.isnan(features['centroid_x'])
        assert np.isnan(features['centroid_y'])
        assert np.isnan(features['centroid_z'])
    
    def test_calculate_cell_features_irregular_shape(self):
        """Test with irregular cell shape."""
        volume = np.zeros((10, 10, 10), dtype=int)
        # Create L-shaped cell
        volume[2:5, 2:8, 2:4] = 1
        volume[2:8, 2:5, 2:4] = 1
        
        features = calculate_cell_features(volume, 1)
        
        # Volume should be correct for L-shape
        assert features['volume'] > 0
        assert features['surface_area'] > 0
        
        # Centroid should be within reasonable bounds
        assert 0 <= features['centroid_x'] <= 10
        assert 0 <= features['centroid_y'] <= 10
        assert 0 <= features['centroid_z'] <= 10


class TestCalculateVelocity:
    """Test calculate_velocity function."""
    
    def test_calculate_velocity_with_previous(self):
        """Test velocity calculation with previous centroid."""
        prev_centroid = np.array([1.0, 2.0, 3.0])
        curr_centroid = np.array([2.0, 4.0, 5.0])
        
        velocity = calculate_velocity(prev_centroid, curr_centroid)
        
        expected_velocity = np.array([1.0, 2.0, 2.0])
        np.testing.assert_array_equal(velocity, expected_velocity)
    
    def test_calculate_velocity_no_previous(self):
        """Test velocity calculation without previous centroid."""
        curr_centroid = np.array([2.0, 4.0, 5.0])
        
        velocity = calculate_velocity(None, curr_centroid)
        
        expected_velocity = np.array([0.0, 0.0, 0.0])
        np.testing.assert_array_equal(velocity, expected_velocity)
    
    def test_calculate_velocity_same_position(self):
        """Test velocity when position doesn't change."""
        centroid = np.array([1.0, 2.0, 3.0])
        
        velocity = calculate_velocity(centroid, centroid)
        
        expected_velocity = np.array([0.0, 0.0, 0.0])
        np.testing.assert_array_equal(velocity, expected_velocity)


class TestCalculateAcceleration:
    """Test calculate_acceleration function."""
    
    def test_calculate_acceleration_with_previous(self):
        """Test acceleration calculation with previous velocity."""
        prev_velocity = np.array([1.0, 1.0, 1.0])
        curr_velocity = np.array([2.0, 3.0, 1.0])
        
        acceleration = calculate_acceleration(prev_velocity, curr_velocity)
        
        expected_acceleration = np.array([1.0, 2.0, 0.0])
        np.testing.assert_array_equal(acceleration, expected_acceleration)
    
    def test_calculate_acceleration_no_previous(self):
        """Test acceleration calculation without previous velocity."""
        curr_velocity = np.array([2.0, 3.0, 1.0])
        
        acceleration = calculate_acceleration(None, curr_velocity)
        
        expected_acceleration = np.array([0.0, 0.0, 0.0])
        np.testing.assert_array_equal(acceleration, expected_acceleration)
    
    def test_calculate_acceleration_constant_velocity(self):
        """Test acceleration when velocity is constant."""
        velocity = np.array([1.0, 2.0, 3.0])
        
        acceleration = calculate_acceleration(velocity, velocity)
        
        expected_acceleration = np.array([0.0, 0.0, 0.0])
        np.testing.assert_array_equal(acceleration, expected_acceleration)


class TestProcessSingleTimepoint:
    """Test process_single_timepoint function."""
    
    @patch('src.features.kinematic_features.get_timepoint_from_filename')
    @patch('src.features.kinematic_features.load_nifti_file')
    @patch('src.features.kinematic_features.get_cell_labels')
    @patch('src.features.kinematic_features.get_cell_id_from_label')
    @patch('src.features.kinematic_features.process_single_cell')
    def test_process_single_timepoint_success(self, mock_process_cell, mock_get_cell_id,
                                              mock_get_labels, mock_load_nifti, 
                                              mock_get_timepoint):
        """Test successful processing of single timepoint."""
        # Setup mocks
        mock_get_timepoint.return_value = "001"
        test_volume = np.random.rand(10, 10, 10)
        mock_load_nifti.return_value = test_volume
        mock_get_labels.return_value = [1, 2, 3]
        mock_get_cell_id.side_effect = ["cell_001", "cell_002", "cell_003"]
        
        # Mock features for each cell
        test_features = [
            {'volume': 100, 'surface_area': 50},
            {'volume': 120, 'surface_area': 60},
            {'volume': 80, 'surface_area': 40}
        ]
        mock_process_cell.side_effect = test_features
        
        # Test processing
        timepoint, processed_cells = process_single_timepoint("test_file.nii.gz")
        
        # Verify results
        assert timepoint == "001"
        assert len(processed_cells) == 3
        
        expected_cells = [
            ("cell_001", test_features[0]),
            ("cell_002", test_features[1]),
            ("cell_003", test_features[2])
        ]
        assert processed_cells == expected_cells
    
    @patch('src.features.kinematic_features.get_timepoint_from_filename')
    @patch('src.features.kinematic_features.load_nifti_file')
    @patch('src.features.kinematic_features.get_cell_labels')
    @patch('src.features.kinematic_features.get_cell_id_from_label')
    @patch('src.features.kinematic_features.process_single_cell')
    def test_process_single_timepoint_with_target_cell(self, mock_process_cell, mock_get_cell_id,
                                                       mock_get_labels, mock_load_nifti,
                                                       mock_get_timepoint):
        """Test processing with target cell filter."""
        # Setup mocks
        mock_get_timepoint.return_value = "001"
        test_volume = np.random.rand(10, 10, 10)
        mock_load_nifti.return_value = test_volume
        mock_get_labels.return_value = [1, 2, 3]
        mock_get_cell_id.side_effect = ["cell_001", "cell_002", "cell_003"]
        
        test_features = {'volume': 100, 'surface_area': 50}
        mock_process_cell.return_value = test_features
        
        # Test processing with target cell
        timepoint, processed_cells = process_single_timepoint("test_file.nii.gz", 
                                                               target_cell="cell_002")
        
        # Should only process cell_002
        assert timepoint == "001"
        assert len(processed_cells) == 1
        assert processed_cells[0] == ("cell_002", test_features)
        
        # process_single_cell should only be called once (for cell_002)
        assert mock_process_cell.call_count == 1
    
    @patch('src.features.kinematic_features.get_timepoint_from_filename')
    @patch('src.features.kinematic_features.load_nifti_file')
    @patch('src.features.kinematic_features.get_cell_labels')
    @patch('src.features.kinematic_features.get_cell_id_from_label')
    @patch('src.features.kinematic_features.process_single_cell')
    def test_process_single_timepoint_with_failed_processing(self, mock_process_cell, mock_get_cell_id,
                                                             mock_get_labels, mock_load_nifti,
                                                             mock_get_timepoint):
        """Test processing when some cells fail to process."""
        # Setup mocks
        mock_get_timepoint.return_value = "001"
        test_volume = np.random.rand(10, 10, 10)
        mock_load_nifti.return_value = test_volume
        mock_get_labels.return_value = [1, 2, 3]
        mock_get_cell_id.side_effect = ["cell_001", "cell_002", "cell_003"]
        
        # Second cell fails to process (returns None)
        test_features = {'volume': 100, 'surface_area': 50}
        mock_process_cell.side_effect = [test_features, None, test_features]
        
        # Test processing
        timepoint, processed_cells = process_single_timepoint("test_file.nii.gz")
        
        # Should only have 2 processed cells (first and third)
        assert timepoint == "001"
        assert len(processed_cells) == 2
        assert processed_cells[0] == ("cell_001", test_features)
        assert processed_cells[1] == ("cell_003", test_features)


class TestExtractCellFeatures:
    """Test extract_cell_features function."""
    
    @patch('src.features.kinematic_features.get_all_nifti_files')
    @patch('src.features.kinematic_features.process_files_parallel')
    @patch('src.features.kinematic_features.save_features')
    @patch('pathlib.Path.mkdir')
    def test_extract_cell_features_success(self, mock_mkdir, mock_save_features,
                                           mock_process_parallel, mock_get_files):
        """Test successful feature extraction."""
        # Setup mocks
        test_files = [Path("file1.nii.gz"), Path("file2.nii.gz")]
        mock_get_files.return_value = test_files
        
        # Mock parallel processing results
        test_results = [
            ("001", [("cell_001", {'volume': 100, 'centroid_x': 1.0, 'centroid_y': 2.0, 'centroid_z': 3.0})]),
            ("002", [("cell_001", {'volume': 110, 'centroid_x': 1.5, 'centroid_y': 2.5, 'centroid_z': 3.5})])
        ]
        mock_process_parallel.return_value = test_results
        
        # Test extraction
        stats = extract_cell_features("test_data_dir", "test_output_dir")
        
        # Verify stats
        assert stats['total_cells'] == 1  # Only one unique cell
        assert stats['processed_timepoints'] == 2
        assert stats['processed_cells'] == {'cell_001'}
        
        # Verify save_features was called with correct parameters
        assert mock_save_features.call_count == 2
        
        # Verify features include velocity and acceleration
        saved_calls = mock_save_features.call_args_list
        first_call_features = saved_calls[0][0][0]  # First argument of first call
        assert 'velocity_x' in first_call_features
        assert 'velocity_y' in first_call_features
        assert 'velocity_z' in first_call_features
        assert 'acceleration_x' in first_call_features
        assert 'acceleration_y' in first_call_features
        assert 'acceleration_z' in first_call_features
    
    @patch('src.features.kinematic_features.get_all_nifti_files')
    def test_extract_cell_features_no_files(self, mock_get_files):
        """Test extraction with no NIfTI files."""
        mock_get_files.return_value = []
        
        with pytest.raises(FileNotFoundError, match="No NIfTI files found"):
            extract_cell_features("empty_dir", "output_dir")
    
    @patch('src.features.kinematic_features.get_all_nifti_files')
    @patch('src.features.kinematic_features.get_timepoint_from_filename')
    def test_extract_cell_features_with_timepoint_filter(self, mock_get_timepoint, mock_get_files):
        """Test extraction with timepoint filtering."""
        # Setup mocks
        test_files = [
            Path("file_001.nii.gz"),
            Path("file_002.nii.gz"),
            Path("file_003.nii.gz")
        ]
        mock_get_files.return_value = test_files
        mock_get_timepoint.side_effect = ["001", "002", "003"]
        
        # Test with timepoint filter
        with pytest.raises(ValueError, match="No NIfTI files found for specified timepoints"):
            extract_cell_features("test_dir", "output_dir", timepoints=["005", "006"])
    
    @patch('src.features.kinematic_features.get_all_nifti_files')
    @patch('src.features.kinematic_features.process_files_parallel')
    @patch('src.features.kinematic_features.save_features')
    @patch('pathlib.Path.mkdir')
    def test_extract_cell_features_velocity_calculation(self, mock_mkdir, mock_save_features,
                                                        mock_process_parallel, mock_get_files):
        """Test that velocity and acceleration are calculated correctly."""
        # Setup mocks
        test_files = [Path("file1.nii.gz"), Path("file2.nii.gz")]
        mock_get_files.return_value = test_files
        
        # Mock two timepoints for same cell with different centroids
        test_results = [
            ("001", [("cell_001", {'volume': 100, 'centroid_x': 1.0, 'centroid_y': 2.0, 'centroid_z': 3.0})]),
            ("002", [("cell_001", {'volume': 110, 'centroid_x': 2.0, 'centroid_y': 3.0, 'centroid_z': 4.0})])
        ]
        mock_process_parallel.return_value = test_results
        
        # Test extraction
        extract_cell_features("test_data_dir", "test_output_dir")
        
        # Get the saved features
        saved_calls = mock_save_features.call_args_list
        
        # First timepoint should have zero velocity
        first_features = saved_calls[0][0][0]
        assert first_features['velocity_x'] == 0.0
        assert first_features['velocity_y'] == 0.0
        assert first_features['velocity_z'] == 0.0
        
        # Second timepoint should have velocity (1, 1, 1)
        second_features = saved_calls[1][0][0]
        assert second_features['velocity_x'] == 1.0
        assert second_features['velocity_y'] == 1.0
        assert second_features['velocity_z'] == 1.0


@pytest.mark.integration
class TestKinematicFeaturesIntegration:
    """Integration tests for kinematic features module."""
    
    def test_feature_calculation_workflow(self):
        """Test the complete feature calculation workflow."""
        # Create a test volume with multiple cells
        volume = np.zeros((20, 20, 20), dtype=int)
        
        # Cell 1: cube at position (5-9, 5-9, 5-9)
        volume[5:10, 5:10, 5:10] = 1
        
        # Cell 2: cube at position (12-15, 12-15, 12-15)
        volume[12:16, 12:16, 12:16] = 2
        
        # Calculate features for both cells
        features_1 = calculate_cell_features(volume, 1)
        features_2 = calculate_cell_features(volume, 2)
        
        # Cell 1 should be larger (5x5x5 = 125 voxels)
        assert features_1['volume'] == 125
        assert abs(features_1['centroid_x'] - 7.0) < 0.1
        
        # Cell 2 should be smaller (4x4x4 = 64 voxels)
        assert features_2['volume'] == 64
        assert abs(features_2['centroid_x'] - 13.5) < 0.1
        
        # Test velocity calculation between timepoints
        centroid_1_t1 = np.array([features_1['centroid_x'], features_1['centroid_y'], features_1['centroid_z']])
        centroid_1_t2 = centroid_1_t1 + np.array([1.0, 0.5, -0.5])  # Simulate movement
        
        velocity = calculate_velocity(centroid_1_t1, centroid_1_t2)
        expected_velocity = np.array([1.0, 0.5, -0.5])
        np.testing.assert_array_almost_equal(velocity, expected_velocity)
        
        # Test acceleration calculation
        velocity_t2 = velocity
        velocity_t3 = velocity + np.array([0.2, -0.1, 0.1])  # Change in velocity
        
        acceleration = calculate_acceleration(velocity_t2, velocity_t3)
        expected_acceleration = np.array([0.2, -0.1, 0.1])
        np.testing.assert_array_almost_equal(acceleration, expected_acceleration)


if __name__ == "__main__":
    pytest.main([__file__])