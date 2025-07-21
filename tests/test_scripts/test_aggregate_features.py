"""
Tests for aggregate_features.py script.
"""

import numpy as np
import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

# Import functions from the main script
# Note: We need to import these carefully due to potential module structure
try:
    from aggregate_features import (
        load_name_dictionary,
        discover_cells_and_timepoints,
        main
    )
except ImportError:
    # Alternative import if the module structure is different
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from aggregate_features import (
        load_name_dictionary,
        discover_cells_and_timepoints,
        main
    )


class TestLoadNameDictionary:
    """Test load_name_dictionary function."""
    
    def test_load_name_dictionary_valid(self):
        """Test loading valid name dictionary."""
        # Create test CSV content
        csv_content = "1,cell_A\n2,cell_B\n3,cell_C\n"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            tmp_file.write(csv_content)
            tmp_file.flush()
            
            try:
                name_to_id_str, id_str_to_name, valid_cell_names = load_name_dictionary(tmp_file.name)
                
                # Verify mappings
                expected_name_to_id = {'cell_A': '001', 'cell_B': '002', 'cell_C': '003'}
                expected_id_to_name = {'001': 'cell_A', '002': 'cell_B', '003': 'cell_C'}
                expected_valid_names = ['cell_A', 'cell_B', 'cell_C']
                
                assert name_to_id_str == expected_name_to_id
                assert id_str_to_name == expected_id_to_name
                assert valid_cell_names == expected_valid_names
                
            finally:
                Path(tmp_file.name).unlink()
    
    def test_load_name_dictionary_with_invalid_entries(self):
        """Test loading name dictionary with some invalid entries."""
        csv_content = "1,cell_A\n,cell_B\n3,cell_C\ninvalid,cell_D\n"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            tmp_file.write(csv_content)
            tmp_file.flush()
            
            try:
                with patch('builtins.print') as mock_print:
                    name_to_id_str, id_str_to_name, valid_cell_names = load_name_dictionary(tmp_file.name)
                
                # Should only include valid entries
                expected_name_to_id = {'cell_A': '001', 'cell_C': '003'}
                expected_valid_names = ['cell_A', 'cell_C']
                
                assert name_to_id_str == expected_name_to_id
                assert valid_cell_names == expected_valid_names
                
                # Should have printed warnings
                assert mock_print.call_count >= 1
                
            finally:
                Path(tmp_file.name).unlink()
    
    def test_load_name_dictionary_large_numbers(self):
        """Test loading name dictionary with large ID numbers."""
        csv_content = "100,cell_X\n999,cell_Y\n1500,cell_Z\n"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            tmp_file.write(csv_content)
            tmp_file.flush()
            
            try:
                name_to_id_str, id_str_to_name, valid_cell_names = load_name_dictionary(tmp_file.name)
                
                # Should handle large numbers correctly
                assert name_to_id_str['cell_X'] == '100'
                assert name_to_id_str['cell_Y'] == '999'
                assert name_to_id_str['cell_Z'] == '1500'  # Should not pad beyond 3 digits for large numbers
                
            finally:
                Path(tmp_file.name).unlink()
    
    def test_load_name_dictionary_float_ids(self):
        """Test loading name dictionary with float ID values."""
        csv_content = "1.0,cell_A\n2.5,cell_B\n3.7,cell_C\n"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            tmp_file.write(csv_content)
            tmp_file.flush()
            
            try:
                name_to_id_str, id_str_to_name, valid_cell_names = load_name_dictionary(tmp_file.name)
                
                # Should convert floats to integers
                expected_name_to_id = {'cell_A': '001', 'cell_B': '002', 'cell_C': '003'}
                assert name_to_id_str == expected_name_to_id
                
            finally:
                Path(tmp_file.name).unlink()
    
    def test_load_name_dictionary_nonexistent_file(self):
        """Test loading non-existent name dictionary file."""
        with patch('builtins.print') as mock_print:
            result = load_name_dictionary("nonexistent_file.csv")
        
        # Should return None values and print error
        assert result == (None, None, None)
        mock_print.assert_called()


class TestDiscoverCellsAndTimepoints:
    """Test discover_cells_and_timepoints function."""
    
    def test_discover_cells_and_timepoints_basic(self):
        """Test basic cell and timepoint discovery."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Create mock directory structure for geo_features
            geo_dir = tmp_path / "geo_features"
            geo_dir.mkdir()
            
            # Create mock cell directories and files
            cell_001_dir = geo_dir / "cell_001"
            cell_001_dir.mkdir()
            (cell_001_dir / "cell_001_010_features.npy").touch()
            (cell_001_dir / "cell_001_020_features.npy").touch()
            
            cell_002_dir = geo_dir / "cell_002"
            cell_002_dir.mkdir()
            (cell_002_dir / "cell_002_010_features.npy").touch()
            (cell_002_dir / "cell_002_030_features.npy").touch()
            
            # Create mock directory structure for spharm
            spharm_dir = tmp_path / "spharm"
            spharm_dir.mkdir()
            
            cell_A_dir = spharm_dir / "cell_A"
            cell_A_dir.mkdir()
            (cell_A_dir / "cell_A_010_l15.npy").touch()
            (cell_A_dir / "cell_A_020_l15.npy").touch()
            
            cell_B_dir = spharm_dir / "cell_B"
            cell_B_dir.mkdir()
            (cell_B_dir / "cell_B_010_l15.npy").touch()
            (cell_B_dir / "cell_B_040_l15.npy").touch()
            
            # Test discovery
            all_cell_names = ['cell_A', 'cell_B', 'cell_C']  # Include cell_C which has no data
            
            with patch('builtins.print'):  # Suppress print statements
                cells, timepoints = discover_cells_and_timepoints(geo_dir, spharm_dir, all_cell_names)
            
            # Should return all cells from dictionary (even those without data)
            assert set(cells) == {'cell_A', 'cell_B', 'cell_C'}
            
            # Should discover all unique timepoints
            expected_timepoints = ['010', '020', '030', '040']
            assert set(timepoints) == set(expected_timepoints)
    
    def test_discover_cells_and_timepoints_empty_directories(self):
        """Test discovery with empty directories."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Create empty directories
            geo_dir = tmp_path / "geo_features"
            spharm_dir = tmp_path / "spharm"
            geo_dir.mkdir()
            spharm_dir.mkdir()
            
            all_cell_names = ['cell_A', 'cell_B']
            
            with patch('builtins.print'):
                cells, timepoints = discover_cells_and_timepoints(geo_dir, spharm_dir, all_cell_names)
            
            # Should still return cell names from dictionary
            assert cells == ['cell_A', 'cell_B']
            
            # Should return empty timepoints
            assert timepoints == []
    
    def test_discover_cells_and_timepoints_nonexistent_directories(self):
        """Test discovery with non-existent directories."""
        nonexistent_geo = Path("nonexistent_geo")
        nonexistent_spharm = Path("nonexistent_spharm")
        
        all_cell_names = ['cell_A', 'cell_B']
        
        with patch('builtins.print'):
            cells, timepoints = discover_cells_and_timepoints(nonexistent_geo, nonexistent_spharm, all_cell_names)
        
        # Should still return cell names from dictionary
        assert cells == ['cell_A', 'cell_B']
        
        # Should return empty timepoints
        assert timepoints == []
    
    def test_discover_cells_and_timepoints_mixed_data(self):
        """Test discovery with cells having different data availability."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Create directories
            geo_dir = tmp_path / "geo_features"
            spharm_dir = tmp_path / "spharm"
            geo_dir.mkdir()
            spharm_dir.mkdir()
            
            # Cell A: only geo data
            cell_001_dir = geo_dir / "cell_001"
            cell_001_dir.mkdir()
            (cell_001_dir / "cell_001_010_features.npy").touch()
            
            # Cell B: only spharm data
            cell_B_dir = spharm_dir / "cell_B"
            cell_B_dir.mkdir()
            (cell_B_dir / "cell_B_020_l15.npy").touch()
            
            # Cell C: both types of data
            cell_002_dir = geo_dir / "cell_002"
            cell_002_dir.mkdir()
            (cell_002_dir / "cell_002_030_features.npy").touch()
            
            cell_C_dir = spharm_dir / "cell_C"
            cell_C_dir.mkdir()
            (cell_C_dir / "cell_C_030_l15.npy").touch()
            
            all_cell_names = ['cell_A', 'cell_B', 'cell_C']
            
            with patch('builtins.print'):
                cells, timepoints = discover_cells_and_timepoints(geo_dir, spharm_dir, all_cell_names)
            
            # Should return all cells
            assert set(cells) == {'cell_A', 'cell_B', 'cell_C'}
            
            # Should discover timepoints from both types of data
            expected_timepoints = ['010', '020', '030']
            assert set(timepoints) == set(expected_timepoints)


class TestMainFunction:
    """Test the main function."""
    
    @patch('aggregate_features.load_name_dictionary')
    @patch('aggregate_features.discover_cells_and_timepoints')
    @patch('numpy.save')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    @patch('builtins.print')
    def test_main_success(self, mock_print, mock_json_dump, mock_file_open,
                          mock_np_save, mock_discover, mock_load_dict):
        """Test successful execution of main function."""
        # Setup mocks
        mock_load_dict.return_value = (
            {'cell_A': '001', 'cell_B': '002'},
            {'001': 'cell_A', '002': 'cell_B'},
            ['cell_A', 'cell_B']
        )
        
        mock_discover.return_value = (['cell_A', 'cell_B'], ['010', '020'])
        
        # Mock feature loading functions
        with patch('aggregate_features.tqdm') as mock_tqdm, \
             patch('numpy.load') as mock_np_load, \
             patch('pathlib.Path.exists') as mock_exists:
            
            # Configure tqdm to act as passthrough
            mock_tqdm.side_effect = lambda x, **kwargs: x
            
            # Mock feature file existence and loading
            mock_exists.return_value = True
            
            # Mock kinematic features
            mock_kinematic_features = {
                'volume': 100,
                'surface_area': 50,
                'centroid_x': 1.0,
                'centroid_y': 2.0,
                'centroid_z': 3.0
            }
            
            # Mock SH coefficients
            mock_sh_coeffs = np.random.rand(225)  # 15^2 = 225 coefficients
            
            mock_np_load.side_effect = lambda path, **kwargs: (
                mock_kinematic_features if 'features.npy' in str(path)
                else mock_sh_coeffs
            )
            
            # Test main function
            main()
            
            # Verify function calls
            mock_load_dict.assert_called_once()
            mock_discover.assert_called_once()
            mock_np_save.assert_called_once()
            mock_json_dump.assert_called_once()
    
    @patch('aggregate_features.load_name_dictionary')
    @patch('builtins.print')
    def test_main_no_name_dictionary(self, mock_print, mock_load_dict):
        """Test main function when name dictionary loading fails."""
        mock_load_dict.return_value = (None, None, None)
        
        # Should return early
        main()
        
        mock_load_dict.assert_called_once()
        # Should print error and return
        mock_print.assert_called()
    
    @patch('aggregate_features.load_name_dictionary')
    @patch('aggregate_features.discover_cells_and_timepoints')
    @patch('builtins.print')
    def test_main_no_cells_or_timepoints(self, mock_print, mock_discover, mock_load_dict):
        """Test main function when no cells or timepoints are discovered."""
        mock_load_dict.return_value = (
            {'cell_A': '001'},
            {'001': 'cell_A'},
            ['cell_A']
        )
        
        # Test with no cells
        mock_discover.return_value = ([], ['010'])
        main()
        
        # Test with no timepoints
        mock_discover.return_value = (['cell_A'], [])
        main()
        
        # Should print error message and return early
        assert mock_print.call_count >= 2


@pytest.mark.integration
class TestAggregateFeatures:
    """Integration tests for aggregate_features module."""
    
    def test_name_dictionary_workflow(self):
        """Test the complete name dictionary workflow."""
        # Create comprehensive test CSV
        csv_content = (
            "1.0,cell_alpha\n"
            "2,cell_beta\n" 
            "10,cell_gamma\n"
            ",invalid_entry\n"
            "invalid,another_invalid\n"
            "50.7,cell_delta\n"
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            tmp_file.write(csv_content)
            tmp_file.flush()
            
            try:
                with patch('builtins.print'):
                    name_to_id_str, id_str_to_name, valid_cell_names = load_name_dictionary(tmp_file.name)
                
                # Should process valid entries correctly
                assert len(valid_cell_names) == 4
                assert 'cell_alpha' in valid_cell_names
                assert 'cell_beta' in valid_cell_names
                assert 'cell_gamma' in valid_cell_names
                assert 'cell_delta' in valid_cell_names
                
                # Check ID formatting
                assert name_to_id_str['cell_alpha'] == '001'
                assert name_to_id_str['cell_beta'] == '002'
                assert name_to_id_str['cell_gamma'] == '010'
                assert name_to_id_str['cell_delta'] == '050'
                
                # Check reverse mapping
                assert id_str_to_name['001'] == 'cell_alpha'
                assert id_str_to_name['010'] == 'cell_gamma'
                
            finally:
                Path(tmp_file.name).unlink()
    
    def test_directory_discovery_workflow(self):
        """Test the complete directory discovery workflow."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Create comprehensive directory structure
            geo_dir = tmp_path / "geo_features"
            spharm_dir = tmp_path / "spharm"
            geo_dir.mkdir()
            spharm_dir.mkdir()
            
            # Create multiple cells with overlapping and unique timepoints
            cells_geo_data = {
                'cell_001': ['005', '010', '015'],
                'cell_002': ['010', '020', '025'],
                'cell_003': ['015', '030']
            }
            
            for cell_id, timepoints in cells_geo_data.items():
                cell_dir = geo_dir / cell_id
                cell_dir.mkdir()
                for tp in timepoints:
                    (cell_dir / f"{cell_id}_{tp}_features.npy").touch()
            
            cells_spharm_data = {
                'cell_A': ['005', '035'],
                'cell_B': ['010', '040'],
                'cell_C': ['025', '045']
            }
            
            for cell_name, timepoints in cells_spharm_data.items():
                cell_dir = spharm_dir / cell_name
                cell_dir.mkdir()
                for tp in timepoints:
                    (cell_dir / f"{cell_name}_{tp}_l15.npy").touch()
            
            # Include cells in dictionary that may not have data files
            all_cell_names = ['cell_A', 'cell_B', 'cell_C', 'cell_D', 'cell_E']
            
            with patch('builtins.print'):
                cells, timepoints = discover_cells_and_timepoints(geo_dir, spharm_dir, all_cell_names)
            
            # Should return all cells from dictionary
            assert set(cells) == set(all_cell_names)
            
            # Should discover all unique timepoints from both directories
            all_expected_timepoints = set()
            for tps in cells_geo_data.values():
                all_expected_timepoints.update(tps)
            for tps in cells_spharm_data.values():
                all_expected_timepoints.update(tps)
            
            assert set(timepoints) == all_expected_timepoints
            
            # Timepoints should be sorted numerically
            assert timepoints == sorted(timepoints, key=int)
    
    @patch('builtins.print')
    def test_error_handling_workflow(self, mock_print):
        """Test error handling in various scenarios."""
        # Test 1: Non-existent name dictionary
        result = load_name_dictionary("completely_fake_file.csv")
        assert result == (None, None, None)
        
        # Test 2: Discovery with non-existent directories
        fake_geo = Path("fake_geo_dir")
        fake_spharm = Path("fake_spharm_dir")
        cells, timepoints = discover_cells_and_timepoints(fake_geo, fake_spharm, ['cell_A'])
        
        assert cells == ['cell_A']  # Should still return dictionary cells
        assert timepoints == []     # Should return empty timepoints
        
        # Verify appropriate warnings were printed
        assert mock_print.call_count >= 2


if __name__ == "__main__":
    pytest.main([__file__])