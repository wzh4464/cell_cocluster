"""
Tests for src.utils.feature_utils module.
"""

import numpy as np
import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock, call

from src.utils.feature_utils import (
    process_files_parallel,
    save_features,
    get_cell_labels,
    process_single_cell
)


class TestProcessFilesParallel:
    """Test process_files_parallel function."""
    
    def dummy_process_func(self, file_path, multiplier=1):
        """Dummy processing function for testing."""
        # Simulate processing by returning file stem times multiplier
        return len(str(file_path.stem)) * multiplier
    
    def dummy_process_func_with_none(self, file_path):
        """Dummy function that returns None for some files."""
        if "skip" in str(file_path):
            return None
        return len(str(file_path.stem))
    
    def dummy_process_func_with_error(self, file_path):
        """Dummy function that raises error for some files."""
        if "error" in str(file_path):
            raise ValueError("Processing error")
        return len(str(file_path.stem))
    
    def test_process_files_parallel_basic(self):
        """Test basic parallel processing functionality."""
        test_files = [
            Path("file1.txt"),
            Path("file2.txt"),
            Path("file3.txt")
        ]
        
        results = process_files_parallel(test_files, self.dummy_process_func)
        
        # Should process all files
        assert len(results) == 3
        expected_results = [5, 5, 5]  # "file1", "file2", "file3" all have 5 chars
        assert results == expected_results
    
    def test_process_files_parallel_with_kwargs(self):
        """Test parallel processing with additional kwargs."""
        test_files = [Path("file1.txt"), Path("file2.txt")]
        
        results = process_files_parallel(test_files, self.dummy_process_func, multiplier=3)
        
        # Should process with multiplier
        expected_results = [15, 15]  # 5 * 3 for each file
        assert results == expected_results
    
    def test_process_files_parallel_with_none_results(self):
        """Test parallel processing filters out None results."""
        test_files = [
            Path("file1.txt"),
            Path("skip_file.txt"),
            Path("file3.txt")
        ]
        
        results = process_files_parallel(test_files, self.dummy_process_func_with_none)
        
        # Should filter out None result from skip_file
        assert len(results) == 2
        assert results == [5, 5]  # Only file1 and file3
    
    def test_process_files_parallel_with_custom_workers(self):
        """Test parallel processing with custom number of workers."""
        test_files = [Path("file1.txt"), Path("file2.txt")]
        
        results = process_files_parallel(test_files, self.dummy_process_func, n_workers=1)
        
        # Should still work with single worker
        assert len(results) == 2
        assert results == [5, 5]
    
    def test_process_files_parallel_empty_list(self):
        """Test parallel processing with empty file list."""
        results = process_files_parallel([], self.dummy_process_func)
        assert results == []


class TestSaveFeatures:
    """Test save_features function."""
    
    def test_save_features_basic(self):
        """Test basic feature saving functionality."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            features = {
                'volume': 100,
                'surface_area': 50,
                'centroid_x': 1.5,
                'centroid_y': 2.5,
                'centroid_z': 3.5
            }
            
            save_features(features, tmp_dir, "cell_001", "010", "kinematic")
            
            # Check that cell directory was created
            cell_dir = Path(tmp_dir) / "cell_001"
            assert cell_dir.exists()
            
            # Check that feature file was created
            feature_file = cell_dir / "cell_001_010_kinematic.npy"
            assert feature_file.exists()
            
            # Check that metadata file was created
            metadata_file = cell_dir / "cell_001_010_kinematic_metadata.json"
            assert metadata_file.exists()
            
            # Verify feature data
            saved_features = np.load(feature_file, allow_pickle=True).item()
            assert saved_features == features
            
            # Verify metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            expected_metadata = {
                'cell_id': 'cell_001',
                'timepoint': '010',
                'feature_names': list(features.keys()),
                'feature_type': 'kinematic'
            }
            assert metadata == expected_metadata
    
    def test_save_features_default_type(self):
        """Test feature saving with default feature type."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            features = {'volume': 100}
            
            save_features(features, tmp_dir, "cell_002", "020")
            
            # Check default feature type is used
            cell_dir = Path(tmp_dir) / "cell_002"
            feature_file = cell_dir / "cell_002_020_features.npy"
            metadata_file = cell_dir / "cell_002_020_features_metadata.json"
            
            assert feature_file.exists()
            assert metadata_file.exists()
    
    def test_save_features_path_object(self):
        """Test feature saving with Path object."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            features = {'volume': 100}
            
            save_features(features, Path(tmp_dir), "cell_003", "030")
            
            cell_dir = Path(tmp_dir) / "cell_003"
            feature_file = cell_dir / "cell_003_030_features.npy"
            assert feature_file.exists()
    
    def test_save_features_existing_directory(self):
        """Test feature saving when cell directory already exists."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create cell directory first
            cell_dir = Path(tmp_dir) / "cell_004"
            cell_dir.mkdir()
            
            features = {'volume': 100}
            save_features(features, tmp_dir, "cell_004", "040")
            
            # Should not raise error and file should exist
            feature_file = cell_dir / "cell_004_040_features.npy"
            assert feature_file.exists()
    
    def test_save_features_complex_data(self):
        """Test feature saving with complex data types."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            features = {
                'volume': 100,
                'centroid': np.array([1.0, 2.0, 3.0]),
                'velocity': np.array([0.5, -0.3, 0.8]),
                'metadata': {'type': 'test_cell'}
            }
            
            save_features(features, tmp_dir, "cell_005", "050")
            
            # Verify saved data
            cell_dir = Path(tmp_dir) / "cell_005"
            feature_file = cell_dir / "cell_005_050_features.npy"
            saved_features = np.load(feature_file, allow_pickle=True).item()
            
            assert saved_features['volume'] == 100
            np.testing.assert_array_equal(saved_features['centroid'], features['centroid'])
            np.testing.assert_array_equal(saved_features['velocity'], features['velocity'])
            assert saved_features['metadata'] == features['metadata']


class TestGetCellLabels:
    """Test get_cell_labels function."""
    
    def test_get_cell_labels_basic(self):
        """Test basic cell label extraction."""
        volume = np.array([[[0, 1, 2],
                           [3, 0, 1],
                           [2, 3, 0]],
                          [[1, 2, 3],
                           [0, 1, 2],
                           [3, 0, 1]]])
        
        labels = get_cell_labels(volume)
        
        # Should return sorted unique labels excluding 0
        expected_labels = np.array([1, 2, 3])
        np.testing.assert_array_equal(labels, expected_labels)
    
    def test_get_cell_labels_only_background(self):
        """Test with volume containing only background."""
        volume = np.zeros((5, 5, 5), dtype=int)
        
        labels = get_cell_labels(volume)
        
        # Should return empty array
        assert len(labels) == 0
    
    def test_get_cell_labels_single_cell(self):
        """Test with volume containing single cell."""
        volume = np.zeros((3, 3, 3), dtype=int)
        volume[1, 1, 1] = 5
        
        labels = get_cell_labels(volume)
        
        expected_labels = np.array([5])
        np.testing.assert_array_equal(labels, expected_labels)
    
    def test_get_cell_labels_non_consecutive(self):
        """Test with non-consecutive cell labels."""
        volume = np.zeros((4, 4, 4), dtype=int)
        volume[0, 0, 0] = 1
        volume[1, 1, 1] = 10
        volume[2, 2, 2] = 5
        
        labels = get_cell_labels(volume)
        
        # Should return sorted labels
        expected_labels = np.array([1, 5, 10])
        np.testing.assert_array_equal(labels, expected_labels)
    
    def test_get_cell_labels_float_volume(self):
        """Test with float volume data."""
        volume = np.array([[[0.0, 1.0, 2.0],
                           [3.0, 0.0, 1.0]]], dtype=float)
        
        labels = get_cell_labels(volume)
        
        expected_labels = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(labels, expected_labels)


class TestProcessSingleCell:
    """Test process_single_cell function."""
    
    def dummy_feature_func(self, volume, label, scale_factor=1.0):
        """Dummy feature calculation function."""
        cell_mask = (volume == label)
        volume_size = np.sum(cell_mask)
        return {
            'volume': volume_size * scale_factor,
            'label': label
        }
    
    def failing_feature_func(self, volume, label):
        """Feature function that always fails."""
        raise ValueError("Feature calculation failed")
    
    def test_process_single_cell_success(self):
        """Test successful single cell processing."""
        volume = np.zeros((5, 5, 5), dtype=int)
        volume[1:4, 1:4, 1:4] = 1  # 3x3x3 cube
        
        result = process_single_cell(volume, 1, self.dummy_feature_func)
        
        assert result is not None
        assert result['volume'] == 27  # 3x3x3 = 27 voxels
        assert result['label'] == 1
    
    def test_process_single_cell_with_kwargs(self):
        """Test single cell processing with additional kwargs."""
        volume = np.zeros((3, 3, 3), dtype=int)
        volume[1, 1, 1] = 2
        
        result = process_single_cell(volume, 2, self.dummy_feature_func, scale_factor=2.0)
        
        assert result is not None
        assert result['volume'] == 2.0  # 1 voxel * 2.0 scale factor
        assert result['label'] == 2
    
    def test_process_single_cell_nonexistent_label(self):
        """Test processing with non-existent cell label."""
        volume = np.zeros((3, 3, 3), dtype=int)
        volume[1, 1, 1] = 1
        
        result = process_single_cell(volume, 999, self.dummy_feature_func)
        
        # Should still work, just return zero volume
        assert result is not None
        assert result['volume'] == 0
        assert result['label'] == 999
    
    def test_process_single_cell_failure(self):
        """Test single cell processing when feature function fails."""
        volume = np.zeros((3, 3, 3), dtype=int)
        volume[1, 1, 1] = 1
        
        with patch('builtins.print') as mock_print:
            result = process_single_cell(volume, 1, self.failing_feature_func)
        
        # Should return None on failure
        assert result is None
        
        # Should print error message
        mock_print.assert_called_once_with("Error processing cell 1: Feature calculation failed")
    
    def test_process_single_cell_empty_volume(self):
        """Test processing with empty volume."""
        volume = np.zeros((2, 2, 2), dtype=int)
        
        result = process_single_cell(volume, 1, self.dummy_feature_func)
        
        assert result is not None
        assert result['volume'] == 0
        assert result['label'] == 1


@pytest.mark.integration
class TestFeatureUtilsIntegration:
    """Integration tests for feature utils module."""
    
    def test_complete_feature_processing_workflow(self):
        """Test the complete feature processing workflow."""
        # Create test volume data
        volume1 = np.zeros((10, 10, 10), dtype=int)
        volume1[2:5, 2:5, 2:5] = 1  # Cell 1
        volume1[6:8, 6:8, 6:8] = 2  # Cell 2
        
        volume2 = np.zeros((10, 10, 10), dtype=int)
        volume2[3:6, 3:6, 3:6] = 1  # Cell 1 moved
        volume2[7:9, 7:9, 7:9] = 2  # Cell 2 moved
        
        def mock_feature_func(volume, label):
            """Mock feature function for integration test."""
            cell_mask = (volume == label)
            if np.sum(cell_mask) == 0:
                return None
            
            # Calculate basic features
            coords = np.where(cell_mask)
            centroid = [np.mean(coords[i]) for i in range(3)]
            
            return {
                'volume': np.sum(cell_mask),
                'centroid_x': centroid[0],
                'centroid_y': centroid[1],
                'centroid_z': centroid[2]
            }
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Process volume 1
            labels1 = get_cell_labels(volume1)
            for label in labels1:
                features = process_single_cell(volume1, label, mock_feature_func)
                if features:
                    save_features(features, tmp_dir, f"cell_{label:03d}", "001")
            
            # Process volume 2
            labels2 = get_cell_labels(volume2)
            for label in labels2:
                features = process_single_cell(volume2, label, mock_feature_func)
                if features:
                    save_features(features, tmp_dir, f"cell_{label:03d}", "002")
            
            # Verify results
            tmp_path = Path(tmp_dir)
            
            # Check cell_001 files
            cell_001_dir = tmp_path / "cell_001"
            assert cell_001_dir.exists()
            assert (cell_001_dir / "cell_001_001_features.npy").exists()
            assert (cell_001_dir / "cell_001_002_features.npy").exists()
            
            # Check cell_002 files
            cell_002_dir = tmp_path / "cell_002"
            assert cell_002_dir.exists()
            assert (cell_002_dir / "cell_002_001_features.npy").exists()
            assert (cell_002_dir / "cell_002_002_features.npy").exists()
            
            # Load and verify features
            features_001_t1 = np.load(cell_001_dir / "cell_001_001_features.npy", allow_pickle=True).item()
            features_001_t2 = np.load(cell_001_dir / "cell_001_002_features.npy", allow_pickle=True).item()
            
            # Cell should have moved from t1 to t2
            assert features_001_t1['volume'] == features_001_t2['volume']  # Same size
            assert features_001_t1['centroid_x'] != features_001_t2['centroid_x']  # Different position
    
    def test_parallel_processing_integration(self):
        """Test parallel processing integration."""
        def mock_process_func(file_path, base_value=10):
            """Mock processing function that simulates file processing."""
            # Simulate different processing times and results
            file_num = int(str(file_path.stem)[-1])
            return base_value + file_num
        
        # Create mock file paths
        test_files = [Path(f"file_{i}.txt") for i in range(5)]
        
        # Test parallel processing
        results = process_files_parallel(test_files, mock_process_func, base_value=100)
        
        # Should have processed all files
        assert len(results) == 5
        expected_results = [100 + i for i in range(5)]
        assert results == expected_results


if __name__ == "__main__":
    pytest.main([__file__])