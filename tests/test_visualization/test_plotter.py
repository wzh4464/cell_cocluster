"""
Tests for src.visualization.plotter module.
"""

import numpy as np
import pytest
import tempfile
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

from src.visualization.plotter import (
    normalize_coordinates,
    visualize_first_frame
)


class TestNormalizeCoordinates:
    """Test normalize_coordinates function."""
    
    def test_normalize_coordinates_basic(self):
        """Test basic coordinate normalization."""
        x = np.array([0, 5, 10])
        y = np.array([2, 4, 6])
        z = np.array([1, 3, 5])
        
        x_norm, y_norm, z_norm = normalize_coordinates(x, y, z)
        
        # Check that normalized coordinates are in [0, 1] range
        assert np.all(x_norm >= 0) and np.all(x_norm <= 1)
        assert np.all(y_norm >= 0) and np.all(y_norm <= 1)
        assert np.all(z_norm >= 0) and np.all(z_norm <= 1)
        
        # Check specific values
        np.testing.assert_array_almost_equal(x_norm, [0.0, 0.5, 1.0])
        np.testing.assert_array_almost_equal(y_norm, [0.0, 0.5, 1.0])
        np.testing.assert_array_almost_equal(z_norm, [0.0, 0.5, 1.0])
    
    def test_normalize_coordinates_single_point(self):
        """Test normalization with single point coordinates."""
        x = np.array([5])
        y = np.array([3])
        z = np.array([7])
        
        # When min == max, normalization should handle division by zero
        with np.errstate(invalid='ignore', divide='ignore'):
            x_norm, y_norm, z_norm = normalize_coordinates(x, y, z)
        
        # Result might be NaN when min == max, which is expected behavior
        assert len(x_norm) == 1
        assert len(y_norm) == 1
        assert len(z_norm) == 1
    
    def test_normalize_coordinates_negative_values(self):
        """Test normalization with negative coordinates."""
        x = np.array([-10, 0, 10])
        y = np.array([-5, 0, 5])
        z = np.array([-2, 1, 4])
        
        x_norm, y_norm, z_norm = normalize_coordinates(x, y, z)
        
        # Check range
        assert np.all(x_norm >= 0) and np.all(x_norm <= 1)
        assert np.all(y_norm >= 0) and np.all(y_norm <= 1)
        assert np.all(z_norm >= 0) and np.all(z_norm <= 1)
        
        # Check specific values
        np.testing.assert_array_almost_equal(x_norm, [0.0, 0.5, 1.0])
        np.testing.assert_array_almost_equal(y_norm, [0.0, 0.5, 1.0])
        np.testing.assert_array_almost_equal(z_norm, [0.0, 1/3, 1.0])
    
    def test_normalize_coordinates_identical_values(self):
        """Test normalization when all coordinates are identical."""
        x = np.array([5, 5, 5])
        y = np.array([3, 3, 3])
        z = np.array([7, 7, 7])
        
        with np.errstate(invalid='ignore', divide='ignore'):
            x_norm, y_norm, z_norm = normalize_coordinates(x, y, z)
        
        # When min == max, result should be all NaN or handled appropriately
        # This is expected behavior for degenerate cases
        assert len(x_norm) == len(x)
        assert len(y_norm) == len(y)
        assert len(z_norm) == len(z)
    
    def test_normalize_coordinates_large_arrays(self):
        """Test normalization with larger arrays."""
        size = 1000
        x = np.random.uniform(-100, 100, size)
        y = np.random.uniform(-50, 150, size)
        z = np.random.uniform(0, 1000, size)
        
        x_norm, y_norm, z_norm = normalize_coordinates(x, y, z)
        
        # Check that all values are in [0, 1] range
        assert np.all(x_norm >= 0) and np.all(x_norm <= 1)
        assert np.all(y_norm >= 0) and np.all(y_norm <= 1)
        assert np.all(z_norm >= 0) and np.all(z_norm <= 1)
        
        # Check that min and max are preserved
        assert np.min(x_norm) == 0.0 or np.isclose(np.min(x_norm), 0.0)
        assert np.max(x_norm) == 1.0 or np.isclose(np.max(x_norm), 1.0)
    
    def test_normalize_coordinates_floating_point(self):
        """Test normalization with floating point coordinates."""
        x = np.array([1.5, 2.7, 3.9])
        y = np.array([0.1, 0.5, 0.9])
        z = np.array([10.2, 15.8, 21.4])
        
        x_norm, y_norm, z_norm = normalize_coordinates(x, y, z)
        
        # Verify normalization
        assert np.all(x_norm >= 0) and np.all(x_norm <= 1)
        assert np.all(y_norm >= 0) and np.all(y_norm <= 1)
        assert np.all(z_norm >= 0) and np.all(z_norm <= 1)
        
        # First and last elements should be 0 and 1 respectively
        np.testing.assert_almost_equal(x_norm[0], 0.0)
        np.testing.assert_almost_equal(x_norm[-1], 1.0)


class TestVisualizeFirstFrame:
    """Test visualize_first_frame function."""
    
    @patch('src.visualization.plotter.get_all_nifti_files')
    @patch('src.visualization.plotter.load_nifti_file')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    def test_visualize_first_frame_success(self, mock_show, mock_savefig, 
                                           mock_load_nifti, mock_get_files):
        """Test successful visualization of first frame."""
        # Setup mocks
        test_files = [Path("file1.nii.gz"), Path("file2.nii.gz")]
        mock_get_files.return_value = test_files
        
        # Create test volume with multiple cells
        test_volume = np.zeros((10, 10, 10), dtype=int)
        test_volume[2:5, 2:5, 2:5] = 1  # Cell 1
        test_volume[6:8, 6:8, 6:8] = 2  # Cell 2
        mock_load_nifti.return_value = test_volume
        
        # Test visualization
        visualize_first_frame("test_dir")
        
        # Verify function calls
        mock_get_files.assert_called_once_with("test_dir")
        mock_load_nifti.assert_called_once_with(str(test_files[0]))
        mock_savefig.assert_called_once_with('first_frame_visualization.png')
        mock_show.assert_called_once()
    
    @patch('src.visualization.plotter.get_all_nifti_files')
    def test_visualize_first_frame_no_files(self, mock_get_files):
        """Test visualization when no NIfTI files are found."""
        mock_get_files.return_value = []
        
        with pytest.raises(FileNotFoundError, match="No NIfTI files found"):
            visualize_first_frame("empty_dir")
    
    @patch('src.visualization.plotter.get_all_nifti_files')
    @patch('src.visualization.plotter.load_nifti_file')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    def test_visualize_first_frame_single_cell(self, mock_show, mock_savefig,
                                               mock_load_nifti, mock_get_files):
        """Test visualization with single cell."""
        # Setup mocks
        test_files = [Path("single_cell.nii.gz")]
        mock_get_files.return_value = test_files
        
        # Create test volume with single cell
        test_volume = np.zeros((5, 5, 5), dtype=int)
        test_volume[1:4, 1:4, 1:4] = 1  # Single cell
        mock_load_nifti.return_value = test_volume
        
        # Test visualization
        visualize_first_frame("test_dir")
        
        # Should complete without error
        mock_savefig.assert_called_once()
        mock_show.assert_called_once()
    
    @patch('src.visualization.plotter.get_all_nifti_files')
    @patch('src.visualization.plotter.load_nifti_file')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    def test_visualize_first_frame_background_only(self, mock_show, mock_savefig,
                                                   mock_load_nifti, mock_get_files):
        """Test visualization with background-only volume."""
        # Setup mocks
        test_files = [Path("background_only.nii.gz")]
        mock_get_files.return_value = test_files
        
        # Create test volume with only background (zeros)
        test_volume = np.zeros((5, 5, 5), dtype=int)
        mock_load_nifti.return_value = test_volume
        
        # Test visualization
        visualize_first_frame("test_dir")
        
        # Should complete without error even with no cells
        mock_savefig.assert_called_once()
        mock_show.assert_called_once()
    
    @patch('src.visualization.plotter.get_all_nifti_files')
    @patch('src.visualization.plotter.load_nifti_file')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    def test_visualize_first_frame_many_cells(self, mock_show, mock_savefig,
                                              mock_load_nifti, mock_get_files):
        """Test visualization with many cells."""
        # Setup mocks
        test_files = [Path("many_cells.nii.gz")]
        mock_get_files.return_value = test_files
        
        # Create test volume with many cells
        test_volume = np.zeros((20, 20, 20), dtype=int)
        for i in range(1, 11):  # 10 cells
            start = i * 2
            test_volume[start:start+1, start:start+1, start:start+1] = i
        mock_load_nifti.return_value = test_volume
        
        # Test visualization
        visualize_first_frame("test_dir")
        
        # Should handle many cells without error
        mock_savefig.assert_called_once()
        mock_show.assert_called_once()
    
    @patch('src.visualization.plotter.get_all_nifti_files')
    @patch('src.visualization.plotter.load_nifti_file')
    def test_visualize_first_frame_load_error(self, mock_load_nifti, mock_get_files):
        """Test visualization when file loading fails."""
        # Setup mocks
        test_files = [Path("corrupt_file.nii.gz")]
        mock_get_files.return_value = test_files
        mock_load_nifti.side_effect = Exception("File corrupted")
        
        # Should propagate the loading error
        with pytest.raises(Exception, match="File corrupted"):
            visualize_first_frame("test_dir")
    
    @patch('src.visualization.plotter.get_all_nifti_files')
    @patch('src.visualization.plotter.load_nifti_file')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    def test_visualize_first_frame_non_consecutive_labels(self, mock_show, mock_savefig,
                                                          mock_load_nifti, mock_get_files):
        """Test visualization with non-consecutive cell labels."""
        # Setup mocks
        test_files = [Path("non_consecutive.nii.gz")]
        mock_get_files.return_value = test_files
        
        # Create test volume with non-consecutive labels
        test_volume = np.zeros((10, 10, 10), dtype=int)
        test_volume[1:3, 1:3, 1:3] = 5   # Cell 5
        test_volume[4:6, 4:6, 4:6] = 10  # Cell 10
        test_volume[7:9, 7:9, 7:9] = 1   # Cell 1
        mock_load_nifti.return_value = test_volume
        
        # Test visualization
        visualize_first_frame("test_dir")
        
        # Should handle non-consecutive labels correctly
        mock_savefig.assert_called_once()
        mock_show.assert_called_once()


@pytest.mark.integration
class TestPlotterIntegration:
    """Integration tests for plotter module."""
    
    def test_normalize_and_visualize_workflow(self):
        """Test the workflow of coordinate normalization in visualization context."""
        # Create test volume data
        test_volume = np.zeros((20, 20, 20), dtype=int)
        
        # Add cells at different positions
        cells_data = [
            (1, (2, 8, 2, 8, 2, 8)),    # Cell 1 at corner
            (2, (10, 15, 10, 15, 10, 15)),  # Cell 2 at middle
            (3, (15, 18, 15, 18, 15, 18))   # Cell 3 at far corner
        ]
        
        for label, (x1, x2, y1, y2, z1, z2) in cells_data:
            test_volume[x1:x2, y1:y2, z1:z2] = label
        
        # Get unique labels (excluding background)
        unique_labels = np.unique(test_volume)[1:]
        assert len(unique_labels) == 3
        
        # Test coordinate normalization for each cell
        for label in unique_labels:
            x, y, z = np.where(test_volume == label)
            
            # Normalize coordinates
            x_norm, y_norm, z_norm = normalize_coordinates(x, y, z)
            
            # Verify normalization
            assert np.all(x_norm >= 0) and np.all(x_norm <= 1)
            assert np.all(y_norm >= 0) and np.all(y_norm <= 1)
            assert np.all(z_norm >= 0) and np.all(z_norm <= 1)
            
            # Verify that coordinates maintain relative positions
            if len(x_norm) > 1:
                assert np.min(x_norm) == 0.0 or np.isclose(np.min(x_norm), 0.0)
                assert np.max(x_norm) == 1.0 or np.isclose(np.max(x_norm), 1.0)
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    def test_complete_visualization_pipeline(self, mock_show, mock_savefig):
        """Test complete visualization pipeline with real data structures."""
        # Create temporary directory structure
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create mock NIfTI files
            nifti_files = [
                Path(tmp_dir) / "WT_Sample1_001_segCell.nii.gz",
                Path(tmp_dir) / "WT_Sample1_002_segCell.nii.gz"
            ]
            
            # Create test volume
            test_volume = np.zeros((15, 15, 15), dtype=int)
            
            # Add multiple cells with different shapes
            test_volume[2:5, 2:5, 2:5] = 1      # Cube cell
            test_volume[8:12, 3:6, 8:11] = 2    # Rectangular cell
            test_volume[6:8, 10:13, 5:8] = 3    # Another cell
            
            with patch('src.visualization.plotter.get_all_nifti_files') as mock_get_files, \
                 patch('src.visualization.plotter.load_nifti_file') as mock_load_nifti:
                
                mock_get_files.return_value = nifti_files
                mock_load_nifti.return_value = test_volume
                
                # Test visualization
                visualize_first_frame(tmp_dir)
                
                # Verify the complete pipeline executed
                mock_get_files.assert_called_once_with(tmp_dir)
                mock_load_nifti.assert_called_once()
                mock_savefig.assert_called_once_with('first_frame_visualization.png')
                mock_show.assert_called_once()
    
    def test_coordinate_normalization_edge_cases(self):
        """Test coordinate normalization with various edge cases."""
        edge_cases = [
            # Case 1: Single point
            ([5], [3], [7]),
            
            # Case 2: Two identical points
            ([5, 5], [3, 3], [7, 7]),
            
            # Case 3: Linear distribution
            ([0, 1, 2, 3, 4], [0, 0, 0, 0, 0], [5, 5, 5, 5, 5]),
            
            # Case 4: Large range
            ([0, 1000], [0, 2000], [0, 500]),
            
            # Case 5: Negative values
            ([-100, -50, 0, 50, 100], [-10, -5, 0, 5, 10], [-1, 0, 1])
        ]
        
        for x, y, z in edge_cases:
            x_arr = np.array(x)
            y_arr = np.array(y)
            z_arr = np.array(z)
            
            with np.errstate(invalid='ignore', divide='ignore'):
                x_norm, y_norm, z_norm = normalize_coordinates(x_arr, y_arr, z_arr)
            
            # Basic checks
            assert len(x_norm) == len(x_arr)
            assert len(y_norm) == len(y_arr)
            assert len(z_norm) == len(z_arr)
            
            # For non-degenerate cases, check range
            if len(np.unique(x_arr)) > 1:
                assert np.all(x_norm >= 0) and np.all(x_norm <= 1)
            if len(np.unique(y_arr)) > 1:
                assert np.all(y_norm >= 0) and np.all(y_norm <= 1)
            if len(np.unique(z_arr)) > 1:
                assert np.all(z_norm >= 0) and np.all(z_norm <= 1)


if __name__ == "__main__":
    pytest.main([__file__])