import numpy as np
import pandas as pd
from scipy import ndimage
from pathlib import Path
from ..data.reader import get_all_nifti_files, load_nifti_file
from ..utils.file_utils import get_timepoint_from_filename, get_cell_id_from_label
from ..utils.feature_utils import (
    process_files_parallel,
    save_features,
    get_cell_labels,
    process_single_cell
)

def calculate_cell_features(volume, label):
    """
    计算单个细胞的形态学特征
    
    Args:
        volume (numpy.ndarray): 3D体积数据，包含多个细胞的标记
        label (int): 目标细胞的标记值
        
    Returns:
        dict: 包含以下特征的字典：
            - volume: 细胞体积（体素数）
            - surface_area: 细胞表面积（体素数）
            - centroid_x/y/z: 细胞质心坐标（像素坐标）
            
    Note:
        所有特征都保存在 features.npy 文件中，元数据保存在 features_metadata.npy 文件中。
        文件保存在 output_dir/cell_id/ 目录下，文件名格式为 {cell_id}_{timepoint}_features.npy。
    """
    # Get cell mask
    cell_mask = (volume == label)
    
    # Calculate basic features
    volume = np.sum(cell_mask)
    surface_area = np.sum(ndimage.binary_dilation(cell_mask) & ~cell_mask)
    
    # Calculate centroid
    y, x, z = np.where(cell_mask)
    centroid = np.array([np.mean(x), np.mean(y), np.mean(z)])
    
    return {
        'volume': volume,
        'surface_area': surface_area,
        'centroid_x': centroid[0],
        'centroid_y': centroid[1],
        'centroid_z': centroid[2]
    }

def calculate_velocity(prev_centroid, curr_centroid):
    """
    计算细胞在两个时间点之间的速度

    Args:
        prev_centroid (numpy.ndarray): 前一个时间点的质心坐标
        curr_centroid (numpy.ndarray): 当前时间点的质心坐标

    Returns:
        numpy.ndarray: 3D速度向量 [vx, vy, vz]
    """
    return np.zeros(3) if prev_centroid is None else curr_centroid - prev_centroid

def calculate_acceleration(prev_velocity, curr_velocity):
    """
    计算细胞在两个时间点之间的加速度

    Args:
        prev_velocity (numpy.ndarray): 前一个时间点的速度向量
        curr_velocity (numpy.ndarray): 当前时间点的速度向量

    Returns:
        numpy.ndarray: 3D加速度向量 [ax, ay, az]
    """
    return np.zeros(3) if prev_velocity is None else curr_velocity - prev_velocity

def process_single_timepoint(file_path, target_cell=None):
    """
    处理单个时间点的所有细胞
    
    Args:
        file_path: NIfTI文件路径
        target_cell: 目标细胞ID，如果为None则处理所有细胞
        
    Returns:
        tuple: (timepoint, processed_cells)
    """
    timepoint = get_timepoint_from_filename(file_path)
    volume = load_nifti_file(file_path)
    labels = get_cell_labels(volume)
    
    processed_cells = []
    for label in labels:
        cell_id = get_cell_id_from_label(int(label))
        
        # 如果指定了目标细胞，只处理该细胞
        if target_cell and cell_id != target_cell:
            continue
            
        # 计算特征
        features = process_single_cell(volume, label, calculate_cell_features)
        if features is None:
            continue
            
        processed_cells.append((cell_id, features))
    
    return timepoint, processed_cells

def extract_cell_features(data_dir, output_dir, target_cell=None, timepoints=None):
    """
    从NIfTI文件提取细胞特征，每个细胞每个时间点保存单独的文件
    
    Args:
        data_dir (str): 包含NIfTI文件的目录路径
        output_dir (str): 输出目录路径
        target_cell (str, optional): 目标细胞ID，如果为None则处理所有细胞
        timepoints (list, optional): 要处理的时间点列表，如果为None则处理所有时间点
        
    Returns:
        dict: 包含处理统计信息的字典
        
    Note:
        对每个细胞每个时间点，会生成两个文件：
        1. features.npy: 包含细胞的基本形态学特征和运动学特征
        2. features_metadata.npy: 包含特征的元数据信息
        
        特征包括：
        - 形态学特征：体积、表面积、质心坐标
        - 运动学特征：速度向量、加速度向量
    """
    nifti_files = get_all_nifti_files(data_dir)
    if not nifti_files:
        raise FileNotFoundError("No NIfTI files found in the data directory")
    
    # 如果指定了时间点，只处理这些时间点
    if timepoints is not None:
        # 确保时间点格式一致（去掉前导零）
        timepoints = [tp.lstrip('0') for tp in timepoints]
        nifti_files = [f for f in nifti_files if get_timepoint_from_filename(f).lstrip('0') in timepoints]
        if not nifti_files:
            raise ValueError(f"No NIfTI files found for specified timepoints: {timepoints}")
    
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 初始化数据结构和统计信息
    cell_trajectories = {}
    stats = {
        'total_cells': 0,
        'processed_timepoints': 0,
        'processed_cells': set()
    }
    
    # 并行处理所有时间点
    results = process_files_parallel(
        nifti_files,
        process_single_timepoint,
        target_cell=target_cell
    )
    
    # 处理结果并计算运动学特征
    for timepoint, processed_cells in results:
        stats['processed_timepoints'] += 1
        
        for cell_id, features in processed_cells:
            curr_centroid = np.array([features['centroid_x'], features['centroid_y'], features['centroid_z']])
            
            # 初始化或更新细胞轨迹
            if cell_id not in cell_trajectories:
                cell_trajectories[cell_id] = {
                    'prev_centroid': None,
                    'prev_velocity': None
                }
                stats['total_cells'] += 1
            
            # 计算速度和加速度
            velocity = calculate_velocity(cell_trajectories[cell_id]['prev_centroid'], curr_centroid)
            acceleration = calculate_acceleration(cell_trajectories[cell_id]['prev_velocity'], velocity)
            
            # 更新轨迹
            cell_trajectories[cell_id]['prev_centroid'] = curr_centroid
            cell_trajectories[cell_id]['prev_velocity'] = velocity
            
            # 添加速度和加速度到特征
            features.update({
                'velocity_x': velocity[0],
                'velocity_y': velocity[1],
                'velocity_z': velocity[2],
                'acceleration_x': acceleration[0],
                'acceleration_y': acceleration[1],
                'acceleration_z': acceleration[2]
            })
            
            # 保存特征
            save_features(features, output_dir, cell_id, timepoint)
            stats['processed_cells'].add(cell_id)
            
            print(f"Processed {cell_id} at timepoint {timepoint}")
    
    return stats

def main():
    """
    主函数，执行特征提取
    """
    data_dir = "DATA/SegmentCellUnified/WT_Sample1LabelUnified"
    output_dir = "DATA/geo_features"
    
    print("Extracting cell features...")
    stats = extract_cell_features(data_dir, output_dir)
    
    print("\nExtraction Statistics:")
    print(f"Total cells processed: {stats['total_cells']}")
    print(f"Timepoints processed: {stats['processed_timepoints']}")
    print(f"Unique cells processed: {len(stats['processed_cells'])}")

if __name__ == "__main__":
    main() 
