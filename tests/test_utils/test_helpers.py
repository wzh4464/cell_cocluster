"""
Tests for src.utils.helpers module.
"""

import numpy as np
import pytest
import tempfile
from pathlib import Path

from src.utils.helpers import (
    ensure_dir,
    save_array,
    load_array,
    calculate_distance
)


class TestEnsureDir:
    """Test ensure_dir function."""
    
    def test_ensure_dir_new_directory(self):
        """Test creating new directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            new_dir = Path(tmp_dir) / "new_directory"
            
            # Directory should not exist initially
            assert not new_dir.exists()
            
            # Create directory
            ensure_dir(new_dir)
            
            # Directory should now exist
            assert new_dir.exists()
            assert new_dir.is_dir()
    
    def test_ensure_dir_existing_directory(self):
        """Test with existing directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            existing_dir = Path(tmp_dir)
            
            # Should not raise error for existing directory
            ensure_dir(existing_dir)
            assert existing_dir.exists()
    
    def test_ensure_dir_nested_directories(self):
        """Test creating nested directories."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            nested_dir = Path(tmp_dir) / "level1" / "level2" / "level3"
            
            # Should create all parent directories
            ensure_dir(nested_dir)
            
            assert nested_dir.exists()
            assert nested_dir.is_dir()
    
    def test_ensure_dir_string_input(self):
        """Test with string input."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            new_dir_str = str(Path(tmp_dir) / "string_dir")
            
            ensure_dir(new_dir_str)
            
            assert Path(new_dir_str).exists()


class TestSaveArray:
    """Test save_array function."""
    
    def test_save_array_basic(self):
        """Test basic array saving."""
        test_array = np.array([1, 2, 3, 4, 5])
        
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp_file:
            save_array(test_array, tmp_file.name)
            
            # Verify file was created
            assert Path(tmp_file.name).exists()
            
            # Load and verify data
            loaded_array = np.load(tmp_file.name)
            np.testing.assert_array_equal(loaded_array, test_array)
            
            # Clean up
            Path(tmp_file.name).unlink()
    
    def test_save_array_2d(self):
        """Test saving 2D array."""
        test_array = np.random.rand(5, 3)
        
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp_file:
            save_array(test_array, tmp_file.name)
            
            loaded_array = np.load(tmp_file.name)
            np.testing.assert_array_equal(loaded_array, test_array)
            
            Path(tmp_file.name).unlink()
    
    def test_save_array_3d(self):
        """Test saving 3D array."""
        test_array = np.random.rand(4, 5, 6)
        
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp_file:
            save_array(test_array, tmp_file.name)
            
            loaded_array = np.load(tmp_file.name)
            np.testing.assert_array_equal(loaded_array, test_array)
            
            Path(tmp_file.name).unlink()
    
    def test_save_array_different_dtypes(self):
        """Test saving arrays with different data types."""
        test_cases = [
            np.array([1, 2, 3], dtype=np.int32),
            np.array([1.1, 2.2, 3.3], dtype=np.float32),
            np.array([1.1, 2.2, 3.3], dtype=np.float64),
            np.array([True, False, True], dtype=bool)
        ]
        
        for test_array in test_cases:
            with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp_file:
                save_array(test_array, tmp_file.name)
                
                loaded_array = np.load(tmp_file.name)
                np.testing.assert_array_equal(loaded_array, test_array)
                assert loaded_array.dtype == test_array.dtype
                
                Path(tmp_file.name).unlink()
    
    def test_save_array_complex_data(self):
        """Test saving complex structured array."""
        # Create structured array
        test_array = np.array([(1, 2.5, 'hello'), (2, 3.14, 'world')],
                             dtype=[('int_field', 'i4'), ('float_field', 'f4'), ('str_field', 'U10')])
        
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp_file:
            save_array(test_array, tmp_file.name)
            
            loaded_array = np.load(tmp_file.name)
            np.testing.assert_array_equal(loaded_array, test_array)
            
            Path(tmp_file.name).unlink()


class TestLoadArray:
    """Test load_array function."""
    
    def test_load_array_basic(self):
        """Test basic array loading."""
        test_array = np.array([1, 2, 3, 4, 5])
        
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp_file:
            np.save(tmp_file.name, test_array)
            
            loaded_array = load_array(tmp_file.name)
            np.testing.assert_array_equal(loaded_array, test_array)
            
            Path(tmp_file.name).unlink()
    
    def test_load_array_nonexistent_file(self):
        """Test loading non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_array("nonexistent_file.npy")
    
    def test_load_array_different_shapes(self):
        """Test loading arrays with different shapes."""
        test_arrays = [
            np.array([1, 2, 3]),  # 1D
            np.array([[1, 2], [3, 4]]),  # 2D
            np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),  # 3D
            np.array(42),  # 0D (scalar)
        ]
        
        for test_array in test_arrays:
            with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp_file:
                np.save(tmp_file.name, test_array)
                
                loaded_array = load_array(tmp_file.name)
                np.testing.assert_array_equal(loaded_array, test_array)
                assert loaded_array.shape == test_array.shape
                
                Path(tmp_file.name).unlink()
    
    def test_load_array_with_pathlib(self):
        """Test loading with Path object."""
        test_array = np.array([1, 2, 3])
        
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp_file:
            np.save(tmp_file.name, test_array)
            
            loaded_array = load_array(Path(tmp_file.name))
            np.testing.assert_array_equal(loaded_array, test_array)
            
            Path(tmp_file.name).unlink()


class TestCalculateDistance:
    """Test calculate_distance function."""
    
    def test_calculate_distance_2d(self):
        """Test distance calculation in 2D."""
        point1 = [0, 0]
        point2 = [3, 4]
        
        distance = calculate_distance(point1, point2)
        expected_distance = 5.0  # 3-4-5 triangle
        
        assert abs(distance - expected_distance) < 1e-10
    
    def test_calculate_distance_3d(self):
        """Test distance calculation in 3D."""
        point1 = [0, 0, 0]
        point2 = [1, 1, 1]
        
        distance = calculate_distance(point1, point2)
        expected_distance = np.sqrt(3)
        
        assert abs(distance - expected_distance) < 1e-10
    
    def test_calculate_distance_same_points(self):
        """Test distance between same points."""
        point = [1, 2, 3]
        
        distance = calculate_distance(point, point)
        assert distance == 0.0
    
    def test_calculate_distance_negative_coordinates(self):
        """Test distance with negative coordinates."""
        point1 = [-1, -2, -3]
        point2 = [1, 2, 3]
        
        distance = calculate_distance(point1, point2)
        expected_distance = np.sqrt(4 + 16 + 36)  # sqrt(2^2 + 4^2 + 6^2)
        
        assert abs(distance - expected_distance) < 1e-10
    
    def test_calculate_distance_numpy_arrays(self):
        """Test distance calculation with numpy arrays."""
        point1 = np.array([0, 0, 0])
        point2 = np.array([3, 4, 0])
        
        distance = calculate_distance(point1, point2)
        expected_distance = 5.0
        
        assert abs(distance - expected_distance) < 1e-10
    
    def test_calculate_distance_mixed_types(self):
        """Test distance calculation with mixed input types."""
        point1 = [1, 2]  # List
        point2 = np.array([4, 6])  # Numpy array
        
        distance = calculate_distance(point1, point2)
        expected_distance = 5.0  # sqrt((4-1)^2 + (6-2)^2) = sqrt(9+16) = 5
        
        assert abs(distance - expected_distance) < 1e-10
    
    def test_calculate_distance_1d(self):
        """Test distance calculation in 1D."""
        point1 = [5]
        point2 = [8]
        
        distance = calculate_distance(point1, point2)
        expected_distance = 3.0
        
        assert abs(distance - expected_distance) < 1e-10
    
    def test_calculate_distance_high_dimensional(self):
        """Test distance calculation in high dimensions."""
        # 5D points
        point1 = [1, 2, 3, 4, 5]
        point2 = [2, 3, 4, 5, 6]
        
        distance = calculate_distance(point1, point2)
        expected_distance = np.sqrt(5)  # sqrt(1^2 + 1^2 + 1^2 + 1^2 + 1^2)
        
        assert abs(distance - expected_distance) < 1e-10
    
    def test_calculate_distance_floating_point(self):
        """Test distance calculation with floating point coordinates."""
        point1 = [1.5, 2.7, 3.1]
        point2 = [4.2, 1.8, 5.6]
        
        distance = calculate_distance(point1, point2)
        
        # Calculate expected distance manually
        dx = 4.2 - 1.5
        dy = 1.8 - 2.7
        dz = 5.6 - 3.1
        expected_distance = np.sqrt(dx*dx + dy*dy + dz*dz)
        
        assert abs(distance - expected_distance) < 1e-10


@pytest.mark.integration
class TestHelpersIntegration:
    """Integration tests for helpers module."""
    
    def test_save_load_workflow(self):
        """Test complete save and load workflow."""
        test_arrays = [
            np.random.rand(10),
            np.random.rand(5, 3),
            np.random.rand(4, 5, 6),
            np.array([1, 2, 3, 4, 5])
        ]
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            ensure_dir(tmp_dir)
            
            for i, test_array in enumerate(test_arrays):
                # Save array
                file_path = Path(tmp_dir) / f"array_{i}.npy"
                save_array(test_array, str(file_path))
                
                # Verify file exists
                assert file_path.exists()
                
                # Load and verify
                loaded_array = load_array(str(file_path))
                np.testing.assert_array_equal(loaded_array, test_array)
    
    def test_directory_and_array_management(self):
        """Test directory creation and array storage workflow."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create nested directory structure
            base_dir = Path(tmp_dir) / "experiment"
            data_dir = base_dir / "data"
            results_dir = base_dir / "results"
            
            for directory in [data_dir, results_dir]:
                ensure_dir(directory)
                assert directory.exists()
            
            # Simulate saving experimental data
            experimental_data = {
                'measurements': np.random.rand(100, 3),
                'timestamps': np.arange(100),
                'metadata': np.array(['exp1', 'exp2', 'exp3'])
            }
            
            for name, data in experimental_data.items():
                file_path = data_dir / f"{name}.npy"
                save_array(data, str(file_path))
            
            # Simulate processing and saving results
            # Load data
            measurements = load_array(str(data_dir / "measurements.npy"))
            timestamps = load_array(str(data_dir / "timestamps.npy"))
            
            # Calculate some results
            mean_measurements = np.mean(measurements, axis=0)
            distance_from_origin = calculate_distance([0, 0, 0], mean_measurements)
            
            # Save results
            save_array(mean_measurements, str(results_dir / "mean_measurements.npy"))
            save_array(np.array([distance_from_origin]), str(results_dir / "distance_metric.npy"))
            
            # Verify all files exist
            expected_files = [
                data_dir / "measurements.npy",
                data_dir / "timestamps.npy", 
                data_dir / "metadata.npy",
                results_dir / "mean_measurements.npy",
                results_dir / "distance_metric.npy"
            ]
            
            for file_path in expected_files:
                assert file_path.exists()
            
            # Verify data integrity
            loaded_mean = load_array(str(results_dir / "mean_measurements.npy"))
            loaded_distance = load_array(str(results_dir / "distance_metric.npy"))
            
            np.testing.assert_array_almost_equal(loaded_mean, mean_measurements)
            np.testing.assert_array_almost_equal(loaded_distance, [distance_from_origin])
    
    def test_distance_calculations_workflow(self):
        """Test various distance calculation scenarios."""
        # Simulate cell positions over time
        time_points = 10
        num_cells = 5
        
        # Generate random cell trajectories
        trajectories = {}
        for cell_id in range(num_cells):
            # Random walk trajectory
            start_pos = np.random.rand(3) * 10
            positions = [start_pos]
            
            for t in range(1, time_points):
                # Random step
                step = (np.random.rand(3) - 0.5) * 2
                new_pos = positions[-1] + step
                positions.append(new_pos)
            
            trajectories[cell_id] = np.array(positions)
        
        # Calculate distances between cells at each time point
        distances_over_time = {}
        
        for t in range(time_points):
            distances_at_t = {}
            
            for cell1 in range(num_cells):
                for cell2 in range(cell1 + 1, num_cells):
                    pos1 = trajectories[cell1][t]
                    pos2 = trajectories[cell2][t]
                    
                    distance = calculate_distance(pos1, pos2)
                    distances_at_t[(cell1, cell2)] = distance
            
            distances_over_time[t] = distances_at_t
        
        # Verify all distances are non-negative
        for t in range(time_points):
            for cell_pair, distance in distances_over_time[t].items():
                assert distance >= 0
        
        # Verify distance symmetry (though we only calculate upper triangle)
        for t in range(time_points):
            for (cell1, cell2), distance in distances_over_time[t].items():
                # Calculate reverse distance
                pos1 = trajectories[cell1][t]
                pos2 = trajectories[cell2][t]
                reverse_distance = calculate_distance(pos2, pos1)
                
                assert abs(distance - reverse_distance) < 1e-10
        
        # Calculate average distances
        avg_distances = {}
        for cell_pair in distances_over_time[0].keys():
            distances = [distances_over_time[t][cell_pair] for t in range(time_points)]
            avg_distances[cell_pair] = np.mean(distances)
        
        # All average distances should be positive
        assert all(dist > 0 for dist in avg_distances.values())


if __name__ == "__main__":
    pytest.main([__file__])