"""
Tests for visualize_lineage.py script.
"""

import numpy as np
import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import functions from the main script
try:
    from visualize_lineage import (
        load_coefficients,
        reconstruct_shape,
        create_animation
    )
except ImportError:
    # Alternative import if the module structure is different
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from visualize_lineage import (
        load_coefficients,
        reconstruct_shape,
        create_animation
    )


class TestLoadCoefficients:
    """Test load_coefficients function."""
    
    def test_load_coefficients_basic(self):
        """Test basic coefficient loading."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Create test coefficient files
            test_data = {
                'MSpppppp_001_l15.npy': np.random.rand(225),
                'MSpppppp_003_l15.npy': np.random.rand(225),
                'MSpppppp_002_l15.npy': np.random.rand(225)
            }
            
            for filename, data in test_data.items():
                np.save(tmp_path / filename, data)
            
            # Test loading
            time_points, coefficients = load_coefficients(tmp_dir)
            
            # Verify results
            assert time_points is not None
            assert coefficients is not None
            assert len(time_points) == 3
            assert len(coefficients) == 3
            
            # Should be sorted by time point
            expected_time_points = np.array([1, 2, 3])
            np.testing.assert_array_equal(time_points, expected_time_points)
            
            # Coefficients should match the sorted order
            assert coefficients.shape == (3, 225)
    
    def test_load_coefficients_empty_directory(self):
        """Test loading from empty directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            time_points, coefficients = load_coefficients(tmp_dir)
            
            # Should return None for empty directory
            assert time_points is None
            assert coefficients is None
    
    def test_load_coefficients_no_npy_files(self):
        """Test loading from directory with no .npy files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Create non-.npy files
            (tmp_path / "test.txt").touch()
            (tmp_path / "data.csv").touch()
            
            time_points, coefficients = load_coefficients(tmp_dir)
            
            # Should return None when no .npy files found
            assert time_points is None
            assert coefficients is None
    
    def test_load_coefficients_invalid_files(self):
        """Test loading with some invalid files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Create valid files
            np.save(tmp_path / "cell_001_l15.npy", np.random.rand(225))
            np.save(tmp_path / "cell_003_l15.npy", np.random.rand(225))
            
            # Create invalid files that can't be parsed
            (tmp_path / "invalid_filename.npy").touch()
            
            with patch('builtins.print'):  # Suppress warning prints
                time_points, coefficients = load_coefficients(tmp_dir)
            
            # Should load valid files and skip invalid ones
            assert time_points is not None
            assert coefficients is not None
            assert len(time_points) == 2
            expected_time_points = np.array([1, 3])
            np.testing.assert_array_equal(time_points, expected_time_points)
    
    def test_load_coefficients_corrupted_file(self):
        """Test loading with corrupted .npy file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Create valid file
            np.save(tmp_path / "cell_001_l15.npy", np.random.rand(225))
            
            # Create corrupted file
            with open(tmp_path / "cell_002_l15.npy", 'w') as f:
                f.write("corrupted data")
            
            with patch('builtins.print'):  # Suppress error prints
                time_points, coefficients = load_coefficients(tmp_dir)
            
            # Should load only the valid file
            assert time_points is not None
            assert coefficients is not None
            assert len(time_points) == 1
            assert time_points[0] == 1
    
    def test_load_coefficients_different_filename_formats(self):
        """Test loading with different filename formats."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Create files with different naming patterns
            test_files = [
                ("prefix_010_suffix.npy", 10),
                ("data_005_end.npy", 5),
                ("cell_100_more.npy", 100)
            ]
            
            for filename, expected_time in test_files:
                np.save(tmp_path / filename, np.random.rand(225))
            
            time_points, coefficients = load_coefficients(tmp_dir)
            
            # Should extract time points correctly
            assert time_points is not None
            assert len(time_points) == 3
            expected_times = np.array([5, 10, 100])  # Should be sorted
            np.testing.assert_array_equal(time_points, expected_times)


class TestReconstructShape:
    """Test reconstruct_shape function."""
    
    def test_reconstruct_shape_basic(self):
        """Test basic shape reconstruction."""
        # Create test coefficients (225 coefficients for l_max=15)
        coefficients = np.random.rand(225) + 1j * np.random.rand(225)
        
        # Create test theta and phi grids
        theta = np.linspace(0, np.pi, 10)
        phi = np.linspace(0, 2*np.pi, 10)
        theta_grid, phi_grid = np.meshgrid(theta, phi)
        
        # Test reconstruction
        shape = reconstruct_shape(coefficients, theta_grid, phi_grid)
        
        # Verify output
        assert shape.shape == theta_grid.shape
        assert shape.dtype == np.float64  # Should be real
        assert np.all(np.isfinite(shape))  # Should not contain NaN or inf
    
    def test_reconstruct_shape_real_coefficients(self):
        """Test reconstruction with real coefficients."""
        # Use real coefficients
        coefficients = np.random.rand(225)
        
        theta = np.linspace(0, np.pi, 5)
        phi = np.linspace(0, 2*np.pi, 8)
        theta_grid, phi_grid = np.meshgrid(theta, phi)
        
        shape = reconstruct_shape(coefficients, theta_grid, phi_grid)
        
        assert shape.shape == (8, 5)  # Note: meshgrid swaps dimensions
        assert np.all(np.isfinite(shape))
    
    def test_reconstruct_shape_zero_coefficients(self):
        """Test reconstruction with zero coefficients."""
        coefficients = np.zeros(225)
        
        theta = np.array([[0, np.pi/2], [np.pi/4, np.pi]])
        phi = np.array([[0, np.pi], [np.pi/2, 2*np.pi]])
        
        shape = reconstruct_shape(coefficients, theta, phi)
        
        # With zero coefficients, result should be all zeros
        np.testing.assert_array_almost_equal(shape, np.zeros_like(theta))
    
    def test_reconstruct_shape_single_coefficient(self):
        """Test reconstruction with only first coefficient non-zero."""
        coefficients = np.zeros(225)
        coefficients[0] = 1.0  # Only l=0, m=0 term
        
        theta = np.linspace(0, np.pi, 3)
        phi = np.linspace(0, 2*np.pi, 4)
        theta_grid, phi_grid = np.meshgrid(theta, phi)
        
        shape = reconstruct_shape(coefficients, theta_grid, phi_grid)
        
        # l=0, m=0 spherical harmonic is constant
        # Y_0^0 = 1/sqrt(4*pi), so result should be approximately constant
        assert np.all(np.isfinite(shape))
        # Check that values are approximately equal (constant function)
        assert np.allclose(shape, shape[0, 0], rtol=1e-10)
    
    def test_reconstruct_shape_different_grid_sizes(self):
        """Test reconstruction with different grid sizes."""
        coefficients = np.random.rand(225)
        
        # Test with different grid sizes
        grid_sizes = [(3, 3), (5, 8), (10, 15), (20, 30)]
        
        for n_theta, n_phi in grid_sizes:
            theta = np.linspace(0, np.pi, n_theta)
            phi = np.linspace(0, 2*np.pi, n_phi)
            theta_grid, phi_grid = np.meshgrid(theta, phi)
            
            shape = reconstruct_shape(coefficients, theta_grid, phi_grid)
            
            assert shape.shape == (n_phi, n_theta)  # meshgrid swaps dimensions
            assert np.all(np.isfinite(shape))


class TestCreateAnimation:
    """Test create_animation function."""
    
    @patch('visualize_lineage.load_coefficients')
    @patch('matplotlib.animation.FuncAnimation')
    @patch('matplotlib.pyplot.close')
    def test_create_animation_success(self, mock_close, mock_animation, mock_load_coeffs):
        """Test successful animation creation."""
        # Setup mocks
        test_time_points = np.array([1, 2, 3])
        test_coefficients = np.random.rand(3, 225)
        mock_load_coeffs.return_value = (test_time_points, test_coefficients)
        
        # Mock animation object
        mock_anim = MagicMock()
        mock_animation.return_value = mock_anim
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Test animation creation
            create_animation(tmp_dir, "test_animation.mp4")
            
            # Verify function calls
            mock_load_coeffs.assert_called_once_with(tmp_dir)
            mock_animation.assert_called_once()
            mock_anim.save.assert_called_once_with("test_animation.mp4", writer='ffmpeg', fps=5)
            mock_close.assert_called_once()
    
    @patch('visualize_lineage.load_coefficients')
    @patch('builtins.print')
    def test_create_animation_no_data(self, mock_print, mock_load_coeffs):
        """Test animation creation with no data."""
        # Mock no data found
        mock_load_coeffs.return_value = (None, None)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Should handle gracefully (implementation dependent)
            try:
                create_animation(tmp_dir, "test_animation.mp4")
            except Exception:
                # May raise exception or handle gracefully - both are valid
                pass
    
    @patch('visualize_lineage.load_coefficients')
    @patch('matplotlib.animation.FuncAnimation')
    @patch('matplotlib.pyplot.close')
    def test_create_animation_single_frame(self, mock_close, mock_animation, mock_load_coeffs):
        """Test animation creation with single frame."""
        # Setup mocks for single frame
        test_time_points = np.array([1])
        test_coefficients = np.random.rand(1, 225)
        mock_load_coeffs.return_value = (test_time_points, test_coefficients)
        
        mock_anim = MagicMock()
        mock_animation.return_value = mock_anim
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            create_animation(tmp_dir, "single_frame.mp4")
            
            # Should still create animation with single frame
            mock_animation.assert_called_once()
            mock_anim.save.assert_called_once()
    
    @patch('visualize_lineage.load_coefficients')
    @patch('matplotlib.animation.FuncAnimation')
    @patch('matplotlib.pyplot.close')
    def test_create_animation_custom_output(self, mock_close, mock_animation, mock_load_coeffs):
        """Test animation creation with custom output filename."""
        # Setup mocks
        test_time_points = np.array([1, 2, 3, 4, 5])
        test_coefficients = np.random.rand(5, 225)
        mock_load_coeffs.return_value = (test_time_points, test_coefficients)
        
        mock_anim = MagicMock()
        mock_animation.return_value = mock_anim
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            custom_output = "my_custom_animation.mp4"
            create_animation(tmp_dir, custom_output)
            
            # Should use custom output filename
            mock_anim.save.assert_called_once_with(custom_output, writer='ffmpeg', fps=5)
    
    @patch('visualize_lineage.load_coefficients')  
    @patch('matplotlib.animation.FuncAnimation')
    def test_create_animation_save_error(self, mock_animation, mock_load_coeffs):
        """Test animation creation when save fails."""
        # Setup mocks
        test_time_points = np.array([1, 2])
        test_coefficients = np.random.rand(2, 225)
        mock_load_coeffs.return_value = (test_time_points, test_coefficients)
        
        # Mock animation that fails to save
        mock_anim = MagicMock()
        mock_anim.save.side_effect = Exception("Save failed")
        mock_animation.return_value = mock_anim
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Should propagate the save error
            with pytest.raises(Exception, match="Save failed"):
                create_animation(tmp_dir, "failing_animation.mp4")


@pytest.mark.integration
class TestVisualizeLineageIntegration:
    """Integration tests for visualize_lineage module."""
    
    def test_complete_workflow(self):
        """Test the complete visualization workflow."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Create realistic test data
            time_points = [1, 5, 10, 15, 20]
            
            for tp in time_points:
                # Create coefficients with some structure
                coefficients = np.zeros(225, dtype=complex)
                
                # Add some low-order harmonics
                coefficients[0] = 1.0  # l=0, m=0 (constant term)
                coefficients[1] = 0.1 * np.random.rand()  # l=1, m=-1
                coefficients[2] = 0.1 * np.random.rand()  # l=1, m=0
                coefficients[3] = 0.1 * np.random.rand()  # l=1, m=1
                
                # Add some noise to higher order terms
                coefficients[4:] = 0.01 * (np.random.rand(221) + 1j * np.random.rand(221))
                
                # Save coefficients
                filename = f"cell_{tp:03d}_l15.npy"
                np.save(tmp_path / filename, coefficients)
            
            # Test loading
            loaded_time_points, loaded_coefficients = load_coefficients(tmp_dir)
            
            # Verify loading worked correctly
            assert loaded_time_points is not None
            assert loaded_coefficients is not None
            assert len(loaded_time_points) == len(time_points)
            np.testing.assert_array_equal(loaded_time_points, time_points)
            assert loaded_coefficients.shape == (5, 225)
            
            # Test reconstruction for each time point
            theta = np.linspace(0, np.pi, 10)
            phi = np.linspace(0, 2*np.pi, 15)
            theta_grid, phi_grid = np.meshgrid(theta, phi)
            
            shapes = []
            for coeffs in loaded_coefficients:
                shape = reconstruct_shape(coeffs, theta_grid, phi_grid)
                shapes.append(shape)
                
                # Verify shape properties
                assert shape.shape == (15, 10)  # phi x theta
                assert np.all(np.isfinite(shape))
            
            # Verify shapes are different (due to different coefficients)
            shapes = np.array(shapes)
            assert shapes.shape == (5, 15, 10)
            
            # Check that shapes vary across time (they should be different)
            # At least some variance should exist
            shape_variance = np.var(shapes, axis=0)
            assert np.any(shape_variance > 1e-10)
    
    def test_coefficient_file_formats(self):
        """Test loading various coefficient file formats."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Test different valid filename formats
            test_cases = [
                ("MSpppppp_001_l15.npy", 1),
                ("cell_010_l15.npy", 10),
                ("data_100_l15.npy", 100),
                ("experiment_005_l15.npy", 5)
            ]
            
            expected_time_points = []
            for filename, time_point in test_cases:
                coefficients = np.random.rand(225) + 1j * np.random.rand(225)
                np.save(tmp_path / filename, coefficients)
                expected_time_points.append(time_point)
            
            # Load and verify
            time_points, coefficients = load_coefficients(tmp_dir)
            
            assert time_points is not None
            assert len(time_points) == len(test_cases)
            
            # Should be sorted
            expected_time_points.sort()
            np.testing.assert_array_equal(time_points, expected_time_points)
    
    def test_reconstruction_mathematical_properties(self):
        """Test mathematical properties of shape reconstruction."""
        # Test with known spherical harmonic coefficients
        coefficients = np.zeros(225, dtype=complex)
        
        # Set Y_0^0 coefficient (constant function)
        coefficients[0] = 1.0 / np.sqrt(4 * np.pi)
        
        # Create grid
        theta = np.linspace(0, np.pi, 20)
        phi = np.linspace(0, 2*np.pi, 30)
        theta_grid, phi_grid = np.meshgrid(theta, phi)
        
        # Reconstruct
        shape = reconstruct_shape(coefficients, theta_grid, phi_grid)
        
        # Y_0^0 should give constant function
        expected_value = 1.0 / np.sqrt(4 * np.pi)
        np.testing.assert_allclose(shape, expected_value, rtol=1e-10)
        
        # Test linearity: reconstruction of sum should equal sum of reconstructions
        coeffs1 = np.zeros(225, dtype=complex)
        coeffs1[0] = 1.0
        
        coeffs2 = np.zeros(225, dtype=complex)
        coeffs2[1] = 0.5
        
        shape1 = reconstruct_shape(coeffs1, theta_grid, phi_grid)
        shape2 = reconstruct_shape(coeffs2, theta_grid, phi_grid)
        shape_sum = reconstruct_shape(coeffs1 + coeffs2, theta_grid, phi_grid)
        
        np.testing.assert_allclose(shape_sum, shape1 + shape2, rtol=1e-10)
    
    @patch('matplotlib.animation.FuncAnimation')
    @patch('matplotlib.pyplot.close')
    def test_animation_workflow(self, mock_close, mock_animation):
        """Test the complete animation creation workflow."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Create a time series of coefficients representing shape evolution
            n_frames = 8
            time_points = list(range(1, n_frames + 1))
            
            for i, tp in enumerate(time_points):
                # Create evolving shape
                coefficients = np.zeros(225, dtype=complex)
                coefficients[0] = 1.0  # Base shape
                
                # Add time-varying components
                phase = 2 * np.pi * i / n_frames
                coefficients[1] = 0.2 * np.exp(1j * phase)  # Rotating component
                coefficients[2] = 0.1 * np.sin(phase)       # Oscillating component
                
                filename = f"evolving_cell_{tp:03d}_l15.npy"
                np.save(tmp_path / filename, coefficients)
            
            # Mock animation creation
            mock_anim = MagicMock()
            mock_animation.return_value = mock_anim
            
            # Test animation creation
            output_file = "evolution_animation.mp4"
            create_animation(tmp_dir, output_file)
            
            # Verify animation was created
            mock_animation.assert_called_once()
            
            # Get the update function that was passed to FuncAnimation
            animation_call_args = mock_animation.call_args
            update_func = animation_call_args[0][1]  # Second argument should be update function
            
            # Test that update function can be called for each frame
            # (This tests the internal animation logic)
            for frame in range(n_frames):
                try:
                    # Should not raise exception
                    update_func(frame)
                except Exception as e:
                    pytest.fail(f"Update function failed for frame {frame}: {e}")
            
            # Verify save was called with correct parameters
            mock_anim.save.assert_called_once_with(output_file, writer='ffmpeg', fps=5)


if __name__ == "__main__":
    pytest.main([__file__])