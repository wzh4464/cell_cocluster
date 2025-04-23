from pathlib import Path
import re
import shutil
from typing import Union, Tuple

def parse_nifti_filename(file_path: Union[str, Path]) -> Tuple[str, str]:
    """
    解析NIfTI文件名，提取胚胎名称和时间点
    
    Args:
        file_path: NIfTI文件路径
        
    Returns:
        tuple: (embryo_name, timepoint)
    """
    file_path = Path(file_path)
    parts = file_path.stem.split('_')

    if len(parts) < 3:
        raise ValueError(f"Invalid NIfTI filename format: {file_path}")
    embryo_name = '_'.join(parts[:-2])  # 获取胚胎名称部分
    return embryo_name, parts[-2]

def get_timepoint_from_filename(file_path: Union[str, Path]) -> str:
    """
    从NIfTI文件名中提取时间点
    
    Args:
        file_path: NIfTI文件路径
        
    Returns:
        str: 时间点
    """
    _, timepoint = parse_nifti_filename(file_path)
    return timepoint

def get_embryo_name_from_filename(file_path):
    """
    从NIfTI文件名中提取胚胎名称
    
    Args:
        file_path (str or Path): NIfTI文件路径
        
    Returns:
        str: 胚胎名称
    """
    embryo_name, _ = parse_nifti_filename(file_path)
    return embryo_name

def get_cell_id_from_label(label: int) -> str:
    """
    从标签值生成细胞ID
    
    Args:
        label: 细胞标签值
        
    Returns:
        str: 细胞ID
    """
    return f"cell_{label:03d}"

def ensure_directory(directory: Union[str, Path]) -> Path:
    """
    确保目录存在，如果不存在则创建
    
    Args:
        directory: 目录路径
        
    Returns:
        Path: 目录的Path对象
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory

def clear_directory(directory: Union[str, Path]) -> None:
    """
    清空目录中的所有内容

    Args:
        directory: 要清空的目录路径
    """
    directory = Path(directory)
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)

def get_file_extension(file_path: Union[str, Path]) -> str:
    """
    获取文件扩展名
    
    Args:
        file_path: 文件路径
        
    Returns:
        str: 文件扩展名（小写）
    """
    return Path(file_path).suffix.lower()

def is_nifti_file(file_path: Union[str, Path]) -> bool:
    """
    检查文件是否为NIfTI文件
    
    Args:
        file_path: 文件路径
        
    Returns:
        bool: 是否为NIfTI文件
    """
    return get_file_extension(file_path) in ['.nii', '.nii.gz']
