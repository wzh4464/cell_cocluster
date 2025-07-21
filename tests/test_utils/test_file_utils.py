"""
Tests for src.utils.file_utils module.
"""

import pytest
import tempfile
import shutil
from pathlib import Path

from src.utils.file_utils import (
    parse_nifti_filename,
    get_timepoint_from_filename,
    get_embryo_name_from_filename,
    get_cell_id_from_label,
    ensure_directory,
    clear_directory,
    get_file_extension,
    is_nifti_file
)


class TestParseNiftiFilename:
    """Test parse_nifti_filename function."""
    
    def test_parse_nifti_filename_basic(self):
        """Test basic NIfTI filename parsing."""
        filename = "WT_Sample1_001_segCell.nii.gz"
        embryo_name, timepoint = parse_nifti_filename(filename)
        
        assert embryo_name == "WT_Sample1"
        assert timepoint == "001"
    
    def test_parse_nifti_filename_complex(self):
        """Test parsing complex filename with underscores."""
        filename = "WT_Sample1_Condition_A_010_segCell.nii.gz"
        embryo_name, timepoint = parse_nifti_filename(filename)
        
        assert embryo_name == "WT_Sample1_Condition_A"
        assert timepoint == "010"
    
    def test_parse_nifti_filename_path_object(self):
        """Test parsing with Path object."""
        filename = Path("WT_Sample1_001_segCell.nii.gz")
        embryo_name, timepoint = parse_nifti_filename(filename)
        
        assert embryo_name == "WT_Sample1"
        assert timepoint == "001"
    
    def test_parse_nifti_filename_with_directory(self):
        """Test parsing filename with directory path."""
        filename = Path("/data/experiments/WT_Sample1_001_segCell.nii.gz")
        embryo_name, timepoint = parse_nifti_filename(filename)
        
        assert embryo_name == "WT_Sample1"
        assert timepoint == "001"
    
    def test_parse_nifti_filename_invalid_format(self):
        """Test parsing with invalid filename format."""
        with pytest.raises(ValueError, match="Invalid NIfTI filename format"):
            parse_nifti_filename("invalid_format.nii.gz")
        
        with pytest.raises(ValueError, match="Invalid NIfTI filename format"):
            parse_nifti_filename("only_one_part.nii.gz")
    
    def test_parse_nifti_filename_edge_cases(self):
        """Test parsing edge cases."""
        # Minimum valid format
        filename = "A_B_C.nii.gz"
        embryo_name, timepoint = parse_nifti_filename(filename)
        assert embryo_name == "A"
        assert timepoint == "B"
        
        # Many underscores
        filename = "A_B_C_D_E_F_G_H.nii.gz"
        embryo_name, timepoint = parse_nifti_filename(filename)
        assert embryo_name == "A_B_C_D_E_F"
        assert timepoint == "G"


class TestGetTimepointFromFilename:
    """Test get_timepoint_from_filename function."""
    
    def test_get_timepoint_basic(self):
        """Test basic timepoint extraction."""
        filename = "WT_Sample1_001_segCell.nii.gz"
        timepoint = get_timepoint_from_filename(filename)
        assert timepoint == "001"
    
    def test_get_timepoint_different_values(self):
        """Test timepoint extraction with different values."""
        test_cases = [
            ("experiment_010_segCell.nii.gz", "010"),
            ("sample_A_B_999_segCell.nii.gz", "999"),
            ("test_000_segCell.nii.gz", "000")
        ]
        
        for filename, expected_timepoint in test_cases:
            timepoint = get_timepoint_from_filename(filename)
            assert timepoint == expected_timepoint
    
    def test_get_timepoint_path_object(self):
        """Test timepoint extraction with Path object."""
        filename = Path("WT_Sample1_001_segCell.nii.gz")
        timepoint = get_timepoint_from_filename(filename)
        assert timepoint == "001"


class TestGetEmbryoNameFromFilename:
    """Test get_embryo_name_from_filename function."""
    
    def test_get_embryo_name_basic(self):
        """Test basic embryo name extraction."""
        filename = "WT_Sample1_001_segCell.nii.gz"
        embryo_name = get_embryo_name_from_filename(filename)
        assert embryo_name == "WT_Sample1"
    
    def test_get_embryo_name_complex(self):
        """Test embryo name extraction with complex names."""
        test_cases = [
            ("Experiment_1_Sample_A_001_segCell.nii.gz", "Experiment_1_Sample_A"),
            ("WT_Control_Group_010_segCell.nii.gz", "WT_Control_Group"),
            ("Mutant_line_123_005_segCell.nii.gz", "Mutant_line_123")
        ]
        
        for filename, expected_embryo in test_cases:
            embryo_name = get_embryo_name_from_filename(filename)
            assert embryo_name == expected_embryo
    
    def test_get_embryo_name_path_object(self):
        """Test embryo name extraction with Path object."""
        filename = Path("WT_Sample1_001_segCell.nii.gz")
        embryo_name = get_embryo_name_from_filename(filename)
        assert embryo_name == "WT_Sample1"


class TestGetCellIdFromLabel:
    """Test get_cell_id_from_label function."""
    
    def test_get_cell_id_integer(self):
        """Test cell ID generation from integer."""
        assert get_cell_id_from_label(1) == "cell_001"
        assert get_cell_id_from_label(10) == "cell_010"
        assert get_cell_id_from_label(999) == "cell_999"
    
    def test_get_cell_id_float(self):
        """Test cell ID generation from float."""
        assert get_cell_id_from_label(1.0) == "cell_001"
        assert get_cell_id_from_label(1.7) == "cell_001"  # Should truncate
        assert get_cell_id_from_label(10.9) == "cell_010"
    
    def test_get_cell_id_large_numbers(self):
        """Test cell ID generation with large numbers."""
        assert get_cell_id_from_label(1000) == "cell_1000"
        assert get_cell_id_from_label(12345) == "cell_12345"
    
    def test_get_cell_id_zero(self):
        """Test cell ID generation for zero."""
        assert get_cell_id_from_label(0) == "cell_000"


class TestEnsureDirectory:
    """Test ensure_directory function."""
    
    def test_ensure_directory_new(self):
        """Test creating new directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            new_dir = Path(tmp_dir) / "new_directory"
            
            # Directory should not exist initially
            assert not new_dir.exists()
            
            # Create directory
            result = ensure_directory(new_dir)
            
            # Directory should now exist
            assert new_dir.exists()
            assert new_dir.is_dir()
            assert result == new_dir
    
    def test_ensure_directory_existing(self):
        """Test with existing directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            existing_dir = Path(tmp_dir)
            
            # Should not raise error for existing directory
            result = ensure_directory(existing_dir)
            assert result == existing_dir
            assert existing_dir.exists()
    
    def test_ensure_directory_nested(self):
        """Test creating nested directories."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            nested_dir = Path(tmp_dir) / "level1" / "level2" / "level3"
            
            # Should create all parent directories
            result = ensure_directory(nested_dir)
            
            assert nested_dir.exists()
            assert nested_dir.is_dir()
            assert result == nested_dir
    
    def test_ensure_directory_string_input(self):
        """Test with string input."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            new_dir_str = str(Path(tmp_dir) / "string_dir")
            
            result = ensure_directory(new_dir_str)
            
            assert Path(new_dir_str).exists()
            assert result == Path(new_dir_str)


class TestClearDirectory:
    """Test clear_directory function."""
    
    def test_clear_directory_with_files(self):
        """Test clearing directory containing files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_dir = Path(tmp_dir) / "test_dir"
            test_dir.mkdir()
            
            # Create some files and subdirectories
            (test_dir / "file1.txt").touch()
            (test_dir / "file2.txt").touch()
            sub_dir = test_dir / "subdir"
            sub_dir.mkdir()
            (sub_dir / "file3.txt").touch()
            
            # Clear directory
            clear_directory(test_dir)
            
            # Directory should exist but be empty
            assert test_dir.exists()
            assert test_dir.is_dir()
            assert list(test_dir.iterdir()) == []
    
    def test_clear_directory_nonexistent(self):
        """Test clearing non-existent directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            nonexistent_dir = Path(tmp_dir) / "nonexistent"
            
            # Should create the directory
            clear_directory(nonexistent_dir)
            
            assert nonexistent_dir.exists()
            assert nonexistent_dir.is_dir()
            assert list(nonexistent_dir.iterdir()) == []
    
    def test_clear_directory_empty(self):
        """Test clearing already empty directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            empty_dir = Path(tmp_dir) / "empty_dir"
            empty_dir.mkdir()
            
            # Should work without error
            clear_directory(empty_dir)
            
            assert empty_dir.exists()
            assert empty_dir.is_dir()
            assert list(empty_dir.iterdir()) == []
    
    def test_clear_directory_string_input(self):
        """Test clearing directory with string input."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_dir_str = str(Path(tmp_dir) / "string_test_dir")
            
            # Create directory and file
            Path(test_dir_str).mkdir()
            (Path(test_dir_str) / "file.txt").touch()
            
            clear_directory(test_dir_str)
            
            assert Path(test_dir_str).exists()
            assert list(Path(test_dir_str).iterdir()) == []


class TestGetFileExtension:
    """Test get_file_extension function."""
    
    def test_get_file_extension_basic(self):
        """Test basic file extension extraction."""
        test_cases = [
            ("file.txt", ".txt"),
            ("data.nii.gz", ".gz"),  # Returns last extension
            ("script.py", ".py"),
            ("image.PNG", ".png"),  # Should be lowercase
            ("document.PDF", ".pdf")
        ]
        
        for filename, expected_ext in test_cases:
            ext = get_file_extension(filename)
            assert ext == expected_ext
    
    def test_get_file_extension_no_extension(self):
        """Test with file having no extension."""
        assert get_file_extension("filename") == ""
        assert get_file_extension("README") == ""
    
    def test_get_file_extension_multiple_dots(self):
        """Test with multiple dots in filename."""
        # Should return the last extension
        assert get_file_extension("file.backup.txt") == ".txt"
        assert get_file_extension("data.tar.gz") == ".gz"
    
    def test_get_file_extension_path_object(self):
        """Test with Path object."""
        file_path = Path("data.nii.gz")
        ext = get_file_extension(file_path)
        assert ext == ".gz"
    
    def test_get_file_extension_with_directory(self):
        """Test with full file path."""
        file_path = "/home/user/data/experiment.nii.gz"
        ext = get_file_extension(file_path)
        assert ext == ".gz"


class TestIsNiftiFile:
    """Test is_nifti_file function."""
    
    def test_is_nifti_file_valid(self):
        """Test with valid NIfTI files."""
        nifti_files = [
            "brain.nii",
            "scan.nii.gz",
            "DATA.NII",  # Case insensitive
            "experiment.NII.GZ",
            Path("data.nii"),
            Path("compressed.nii.gz")
        ]
        
        for filename in nifti_files:
            assert is_nifti_file(filename) is True
    
    def test_is_nifti_file_invalid(self):
        """Test with non-NIfTI files."""
        non_nifti_files = [
            "document.txt",
            "image.png",
            "data.csv",
            "script.py",
            "archive.tar.gz",
            "config.json",
            Path("video.mp4"),
            Path("data.xlsx")
        ]
        
        for filename in non_nifti_files:
            assert is_nifti_file(filename) is False
    
    def test_is_nifti_file_edge_cases(self):
        """Test edge cases."""
        edge_cases = [
            ("file.nii.txt", False),  # Has .nii but ends with .txt
            ("data.gz", False),       # Has .gz but not .nii.gz
            ("test.nii.backup", False),  # Has .nii but ends differently
            ("", False),              # Empty string
            ("file", False),          # No extension
            (".nii", True),           # Just extension
            (".nii.gz", True)         # Just extension
        ]
        
        for filename, expected in edge_cases:
            assert is_nifti_file(filename) == expected


@pytest.mark.integration
class TestFileUtilsIntegration:
    """Integration tests for file utils module."""
    
    def test_filename_parsing_workflow(self):
        """Test complete filename parsing workflow."""
        test_filenames = [
            "WT_Sample1_001_segCell.nii.gz",
            "Mutant_Line_A_010_segCell.nii.gz", 
            "Control_Group_B_999_segCell.nii.gz"
        ]
        
        for filename in test_filenames:
            # Parse filename
            embryo_name, timepoint = parse_nifti_filename(filename)
            
            # Extract components individually
            extracted_embryo = get_embryo_name_from_filename(filename)
            extracted_timepoint = get_timepoint_from_filename(filename)
            
            # Should be consistent
            assert embryo_name == extracted_embryo
            assert timepoint == extracted_timepoint
            
            # Should be valid NIfTI file
            assert is_nifti_file(filename) is True
    
    def test_directory_management_workflow(self):
        """Test complete directory management workflow."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_dir = Path(tmp_dir)
            
            # Create a complex directory structure
            project_dir = base_dir / "project"
            ensure_directory(project_dir)
            
            # Create subdirectories for different cell types
            for cell_type in ["cell_001", "cell_002", "cell_003"]:
                cell_dir = project_dir / cell_type
                ensure_directory(cell_dir)
                
                # Create some test files
                for timepoint in ["001", "002", "003"]:
                    feature_file = cell_dir / f"{cell_type}_{timepoint}_features.npy"
                    feature_file.touch()
            
            # Verify structure was created
            assert project_dir.exists()
            assert len(list(project_dir.iterdir())) == 3
            
            # Clear one cell directory
            cell_001_dir = project_dir / "cell_001"
            clear_directory(cell_001_dir)
            
            # Should be empty but still exist
            assert cell_001_dir.exists()
            assert len(list(cell_001_dir.iterdir())) == 0
            
            # Other directories should be untouched
            cell_002_dir = project_dir / "cell_002"
            assert len(list(cell_002_dir.iterdir())) == 3
    
    def test_file_processing_simulation(self):
        """Test file processing simulation using utils."""
        # Simulate processing multiple NIfTI files
        test_files = [
            "WT_Sample1_001_segCell.nii.gz",
            "WT_Sample1_002_segCell.nii.gz",
            "WT_Sample1_003_segCell.nii.gz"
        ]
        
        processing_results = []
        
        for filename in test_files:
            # Extract information
            embryo_name = get_embryo_name_from_filename(filename)
            timepoint = get_timepoint_from_filename(filename)
            ext = get_file_extension(filename)
            is_nifti = is_nifti_file(filename)
            
            # Simulate finding cells in the file
            simulated_labels = [1, 2, 3]  # Simulated cell labels
            
            for label in simulated_labels:
                cell_id = get_cell_id_from_label(label)
                
                processing_results.append({
                    'filename': filename,
                    'embryo_name': embryo_name,
                    'timepoint': timepoint,
                    'cell_id': cell_id,
                    'is_nifti': is_nifti,
                    'extension': ext
                })
        
        # Verify results
        assert len(processing_results) == 9  # 3 files * 3 cells each
        
        # All should be NIfTI files
        assert all(result['is_nifti'] for result in processing_results)
        
        # All should have same embryo name
        embryo_names = set(result['embryo_name'] for result in processing_results)
        assert embryo_names == {"WT_Sample1"}
        
        # Should have all timepoints
        timepoints = set(result['timepoint'] for result in processing_results)
        assert timepoints == {"001", "002", "003"}
        
        # Should have all cell IDs
        cell_ids = set(result['cell_id'] for result in processing_results)
        assert cell_ids == {"cell_001", "cell_002", "cell_003"}


if __name__ == "__main__":
    pytest.main([__file__])