from pathlib import Path
import re

def parse_nifti_filename(file_path):
    """
    解析NIfTI文件名，提取胚胎名称和时间点
    
    Args:
        file_path (str or Path): NIfTI文件路径
        
    Returns:
        tuple: (embryo_name, timepoint)
    """
    file_path = Path(file_path)
    parts = file_path.stem.split('_')

    if len(parts) < 3:
        raise ValueError(f"Invalid NIfTI filename format: {file_path}")
    embryo_name = '_'.join(parts[:-2])  # 获取胚胎名称部分
    return embryo_name, parts[-2]

def get_timepoint_from_filename(file_path):
    """
    从NIfTI文件名中提取时间点
    
    Args:
        file_path (str or Path): NIfTI文件路径
        
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

def get_cell_id_from_label(label):
    """
    将细胞标签转换为细胞ID
    
    Args:
        label (int): 细胞标签
        
    Returns:
        str: 细胞ID (格式: "cell_{label}")
    """
    return f"cell_{label}"
