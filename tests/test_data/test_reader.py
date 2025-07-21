"""
Tests for src.data.reader module.
"""

import numpy as np
import pytest
import tempfile
import nibabel as nib
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.data.reader import (
    load_nifti_file,
    save_nifti_file, 
    get_all_nifti_files,
    process_nifti_files
)


class TestLoadNiftiFile:
    """Test load_nifti_file function."""
    
    def test_load_nifti_file_success(self):
        """Test successful loading of a NIfTI file."""
        # Create a temporary NIfTI file
        test_data = np.random.rand(10, 10, 10)
        img = nib.Nifti1Image(test_data, np.eye(4))
        
        with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as tmp_file:
            nib.save(img, tmp_file.name)
            
            # Test loading
            loaded_data = load_nifti_file(tmp_file.name)
            
            # Verify data matches
            np.testing.assert_array_almost_equal(loaded_data, test_data)
            
            # Clean up
            Path(tmp_file.name).unlink()
    
    def test_load_nifti_file_not_found(self):
        """Test FileNotFoundError when file doesn't exist."""
        with pytest.raises(FileNotFoundError, match="File not found"):
            load_nifti_file("nonexistent_file.nii.gz")
    
    def test_load_nifti_file_with_pathlib(self):
        """Test loading with Path object."""
        test_data = np.random.rand(5, 5, 5)
        img = nib.Nifti1Image(test_data, np.eye(4))
        
        with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as tmp_file:
            nib.save(img, tmp_file.name)
            
            # Test loading with Path object
            loaded_data = load_nifti_file(Path(tmp_file.name))
            np.testing.assert_array_almost_equal(loaded_data, test_data)
            
            # Clean up
            Path(tmp_file.name).unlink()


class TestSaveNiftiFile:
    """Test save_nifti_file function."""
    
    def test_save_nifti_file_success(self):
        """Test successful saving of a NIfTI file."""
        test_data = np.random.rand(8, 8, 8)
        
        with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as tmp_file:
            # Save the data
            save_nifti_file(test_data, tmp_file.name)
            
            # Verify file was created and data matches
            assert Path(tmp_file.name).exists()
            loaded_data = load_nifti_file(tmp_file.name)
            np.testing.assert_array_almost_equal(loaded_data, test_data)
            
            # Clean up
            Path(tmp_file.name).unlink()
    
    def test_save_nifti_file_with_pathlib(self):
        """Test saving with Path object."""
        test_data = np.random.rand(6, 6, 6)
        
        with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as tmp_file:
            # Save with Path object
            save_nifti_file(test_data, Path(tmp_file.name))
            
            # Verify
            assert Path(tmp_file.name).exists()
            loaded_data = load_nifti_file(tmp_file.name)
            np.testing.assert_array_almost_equal(loaded_data, test_data)
            
            # Clean up
            Path(tmp_file.name).unlink()


class TestGetAllNiftiFiles:
    """Test get_all_nifti_files function."""
    
    def test_get_all_nifti_files_empty_directory(self):
        """Test with empty directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            files = get_all_nifti_files(tmp_dir)
            assert files == []
    
    def test_get_all_nifti_files_with_files(self):
        """Test with directory containing NIfTI files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Create test files
            test_files = [
                "WT_Sample1_001_segCell.nii.gz",
                "WT_Sample1_002_segCell.nii.gz", 
                "WT_Sample1_003_segCell.nii.gz",
                "other_file.txt"  # Should be ignored
            ]
            
            for filename in test_files:
                (tmp_path / filename).touch()
            
            # Get NIfTI files
            nifti_files = get_all_nifti_files(tmp_dir)
            
            # Should return only the _segCell.nii.gz files, sorted
            expected_files = [
                tmp_path / "WT_Sample1_001_segCell.nii.gz",
                tmp_path / "WT_Sample1_002_segCell.nii.gz",
                tmp_path / "WT_Sample1_003_segCell.nii.gz"
            ]
            
            assert nifti_files == expected_files
    
    def test_get_all_nifti_files_with_pathlib(self):
        """Test with Path object."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Create a test file
            test_file = tmp_path / "test_001_segCell.nii.gz"
            test_file.touch()
            
            # Test with Path object
            nifti_files = get_all_nifti_files(tmp_path)
            assert nifti_files == [test_file]


class TestProcessNiftiFiles:
    """Test process_nifti_files function."""
    
    @patch('src.data.reader.get_all_nifti_files')
    @patch('src.data.reader.load_nifti_file')
    @patch('numpy.save')
    @patch('pathlib.Path.mkdir')
    def test_process_nifti_files_success(self, mock_mkdir, mock_np_save, 
                                         mock_load_nifti, mock_get_files):
        """Test successful processing of NIfTI files."""
        # Setup mocks
        test_files = [
            Path("WT_Sample1_001_segCell.nii.gz"),
            Path("WT_Sample1_002_segCell.nii.gz")
        ]
        mock_get_files.return_value = test_files
        
        # Mock volume data
        test_volume_1 = np.random.rand(10, 10, 10)
        test_volume_2 = np.random.rand(10, 10, 10)
        mock_load_nifti.side_effect = [test_volume_1, test_volume_2]
        
        # Test processing
        volumes = process_nifti_files("test_dir")
        
        # Verify results
        assert len(volumes) == 2
        assert "001" in volumes
        assert "002" in volumes
        np.testing.assert_array_equal(volumes["001"], test_volume_1)
        np.testing.assert_array_equal(volumes["002"], test_volume_2)
        
        # Verify mkdir was called
        mock_mkdir.assert_called_once_with(exist_ok=True)
        
        # Verify numpy save was called
        assert mock_np_save.call_count == 2
    
    @patch('src.data.reader.get_all_nifti_files')
    @patch('src.data.reader.load_nifti_file')
    def test_process_nifti_files_with_error(self, mock_load_nifti, mock_get_files):
        """Test processing with file loading error."""
        # Setup mocks
        test_files = [
            Path("WT_Sample1_001_segCell.nii.gz"),
            Path("WT_Sample1_002_segCell.nii.gz")
        ]
        mock_get_files.return_value = test_files
        
        # First file loads successfully, second fails
        test_volume = np.random.rand(10, 10, 10)
        mock_load_nifti.side_effect = [test_volume, Exception("Load error")]
        
        # Test processing (should not raise exception)
        with patch('builtins.print') as mock_print:
            volumes = process_nifti_files("test_dir")
        
        # Should only have one volume (the successful one)
        assert len(volumes) == 1
        assert "001" in volumes
        
        # Should have printed error message
        mock_print.assert_any_call("Error processing WT_Sample1_002_segCell.nii.gz: Load error")
    
    @patch('src.data.reader.get_all_nifti_files')
    def test_process_nifti_files_no_files(self, mock_get_files):
        """Test processing with no NIfTI files."""
        mock_get_files.return_value = []
        
        volumes = process_nifti_files("empty_dir")
        assert volumes == {}


@pytest.mark.integration
class TestReaderIntegration:
    """Integration tests for the reader module."""
    
    def test_full_workflow(self):
        """Test the complete workflow of saving and loading."""
        # Create test data
        original_data = np.random.rand(12, 12, 12)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Save the data
            file_path = Path(tmp_dir) / "test_data.nii.gz"
            save_nifti_file(original_data, file_path)
            
            # Load it back
            loaded_data = load_nifti_file(file_path)
            
            # Verify they match
            np.testing.assert_array_almost_equal(loaded_data, original_data)
    
    def test_directory_processing_workflow(self):
        """Test directory processing with real files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Create test NIfTI files
            test_data_1 = np.random.rand(8, 8, 8)
            test_data_2 = np.random.rand(8, 8, 8)
            
            file1 = tmp_path / "WT_Sample1_001_segCell.nii.gz"
            file2 = tmp_path / "WT_Sample1_002_segCell.nii.gz"
            
            save_nifti_file(test_data_1, file1)
            save_nifti_file(test_data_2, file2)
            
            # Process directory
            with patch('pathlib.Path.mkdir'):  # Prevent creating data_npy in test
                volumes = process_nifti_files(tmp_dir)
            
            # Verify results
            assert len(volumes) == 2
            assert "001" in volumes
            assert "002" in volumes
            np.testing.assert_array_almost_equal(volumes["001"], test_data_1)
            np.testing.assert_array_almost_equal(volumes["002"], test_data_2)


if __name__ == "__main__":
    pytest.main([__file__])