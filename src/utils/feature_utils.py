import numpy as np
from pathlib import Path
import multiprocessing as mp
from functools import partial
from typing import List, Dict, Any, Callable, Union
import json
from ..data.reader import load_nifti_file

def process_files_parallel(
    file_paths: List[Path],
    process_func: Callable,
    n_workers: int = None,
    **kwargs
) -> List[Any]:
    """
    并行处理文件
    
    Args:
        file_paths: 要处理的文件路径列表
        process_func: 处理单个文件的函数
        n_workers: 并行工作进程数，默认为CPU核心数
        **kwargs: 传递给process_func的额外参数
        
    Returns:
        处理结果的列表
    """
    if n_workers is None:
        n_workers = min(40, mp.cpu_count())

    with mp.Pool(n_workers) as pool:
        process_func_with_args = partial(process_func, **kwargs)
        results = pool.map(process_func_with_args, file_paths)

    return [r for r in results if r is not None]

def save_features(
    features: Dict[str, Any],
    output_dir: Union[str, Path],
    cell_id: str,
    timepoint: str,
    feature_type: str = "features"
) -> None:
    """
    保存特征数据和元数据
    
    Args:
        features: 特征字典，包含细胞的各种特征值
        output_dir: 输出目录
        cell_id: 细胞ID
        timepoint: 时间点
        feature_type: 特征类型（用于文件名）
        
    Note:
        保存两个文件：
        1. {cell_id}_{timepoint}_{feature_type}.npy: 包含细胞的所有特征值（numpy数组）
        2. {cell_id}_{timepoint}_{feature_type}_metadata.json: 包含特征的元数据信息（JSON格式）
        
        文件保存在 output_dir/cell_id/ 目录下。
    """
    output_dir = Path(output_dir)
    cell_dir = output_dir / cell_id
    cell_dir.mkdir(exist_ok=True)
    
    # 保存特征
    feature_file = cell_dir / f"{cell_id}_{timepoint}_{feature_type}.npy"
    np.save(feature_file, features)
    
    # 保存元数据为JSON
    metadata = {
        'cell_id': cell_id,
        'timepoint': timepoint,
        'feature_names': list(features.keys()),
        'feature_type': feature_type
    }
    metadata_file = cell_dir / f"{cell_id}_{timepoint}_{feature_type}_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=4)

def get_cell_labels(volume: np.ndarray) -> np.ndarray:
    """
    从体积数据中获取所有细胞标签（排除背景）
    
    Args:
        volume: 体积数据数组
        
    Returns:
        细胞标签数组
    """
    return np.unique(volume)[1:]  # 跳过背景标签0

def process_single_cell(
    volume: np.ndarray,
    label: int,
    feature_func: Callable,
    **kwargs
) -> Dict[str, Any]:
    """
    处理单个细胞
    
    Args:
        volume: 体积数据数组
        label: 细胞标签
        feature_func: 计算特征的函数
        **kwargs: 传递给feature_func的额外参数
        
    Returns:
        细胞特征字典
    """
    try:
        return feature_func(volume, label, **kwargs)
    except Exception as e:
        print(f"Error processing cell {label}: {str(e)}")
        return None 
