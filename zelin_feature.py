###
# File: ./zelin_feature.py
# Created Date: Tuesday, April 8th 2025
# Author: Zihan
# -----
# Last Modified: Wednesday, 23rd April 2025 7:00:24 pm
# Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
# -----
# HISTORY:
# Date      		By   	Comments
# ----------		------	---------------------------------------------------------
###

import os
import numpy as np
import nibabel as nib
from pathlib import Path
import multiprocessing as mp
from functools import partial

# Import 3DCSQ functions
import threeDCSQ.utils.cell_func as cell_f
import threeDCSQ.utils.general_func as general_f
from threeDCSQ.transformation.SH_represention import sample_and_SHc_with_surface
from threeDCSQ.analysis.SH_analyses import analysis_SHcPCA_One_embryo
from threeDCSQ.static import config
from threeDCSQ.transformation.SH_represention import get_SH_coefficient_of_embryo

# Additional imports for parallel SH coefficient calculation and npy saving
from multiprocessing import Pool
import pandas as pd
from tqdm import tqdm


def process_single_frame(file_path, embryo_name):
    """Process a single frame file."""
    try:
        # Extract the frame number
        frame_num = file_path.stem.split("_")[-2]

        # Create a Nifti1Image and save it in expected format
        img_data = nib.load(str(file_path)).get_fdata()
        img = nib.Nifti1Image(img_data, np.eye(4))

        # Path where 3DCSQ expects to find the file
        target_path = f"DATA/SegmentCellUnified/{embryo_name}/{embryo_name}_{frame_num}_segCell.nii.gz"
        nib.save(img, target_path)

        print(f"Saved frame {frame_num} for processing")
        return True
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return False


def load_and_process_embryo(data_dir):
    """Load and process one embryo's time series for 3DCSQ analysis."""
    # Directory containing the NIfTI files
    nifti_files = sorted(Path(data_dir).glob("WT_Sample1_*_segCell.nii.gz"))

    # Create directory structure expected by 3DCSQ
    embryo_name = "WT_Sample1LabelUnified"
    os.makedirs(f"DATA/SegmentCellUnified/{embryo_name}", exist_ok=True)

    # Process files in parallel
    num_cpus = min(20, mp.cpu_count())  # Use up to 20 CPUs
    with mp.Pool(num_cpus) as pool:
        process_func = partial(process_single_frame, embryo_name=embryo_name)
        results = pool.map(process_func, nifti_files)

    return f"DATA/SegmentCellUnified/{embryo_name}"


def process_single_frame_volume_surface(frame_path, name_dict_path):
    """Process volume and surface for a single frame."""
    try:
        # Load the image
        img = general_f.load_nitf2_img(frame_path)

        # Calculate volume and surface
        volume_cnt, surface_cnt = cell_f.nii_count_volume_surface(img)

        # Get frame number from filename
        frame_num = os.path.basename(frame_path).split("_")[-2]

        # Create a DataFrame for this frame
        import pandas as pd

        data = []
        for cell_id, volume in volume_cnt.items():
            if cell_id in name_dict_path:
                cell_name = name_dict_path[cell_id]
                surface = surface_cnt.get(cell_id, 0)
                normalized_c = (volume / 10000) ** (1 / 3)
                data.append(
                    {
                        "frame": frame_num,
                        "cell_id": cell_id,
                        "cell_name": cell_name,
                        "volume": volume,
                        "surface": surface,
                        "normalized_c": normalized_c,
                    }
                )

        return pd.DataFrame(data)
    except Exception as e:
        print(f"Error processing {frame_path}: {str(e)}")
        return None


def calculate_volume_surface(embryo_path):
    """Calculate volume and surface area for all cells in the embryo."""
    print("Calculating volume and surface area...")

    # Get the name dictionary path
    name_dict_path = os.path.join(
        config.win_cell_shape_analysis_data_path, "name_dictionary.csv"
    )

    # Get cell name mapping
    name_dict, _ = cell_f.get_cell_name_affine_table(name_dict_path)

    # Get all frame files
    frame_files = sorted(
        [
            os.path.join(embryo_path, f)
            for f in os.listdir(embryo_path)
            if f.endswith("_segCell.nii.gz")
        ]
    )

    # Process files in parallel
    num_cpus = min(20, mp.cpu_count())  # Use up to 20 CPUs
    with mp.Pool(num_cpus) as pool:
        process_func = partial(
            process_single_frame_volume_surface, name_dict_path=name_dict
        )
        results = pool.map(process_func, frame_files)

    # Combine results
    import pandas as pd

    all_data = pd.concat([df for df in results if df is not None], ignore_index=True)

    # Save to CSV
    output_path = os.path.join(embryo_path, "volume_surface_data.csv")
    all_data.to_csv(output_path, index=False)

    print(f"Volume and surface calculation completed. Results saved to {output_path}")


def calculate_spherical_harmonics(embryo_path, l_degree=25):
    """Calculate spherical harmonic coefficients for all cells in the embryo."""
    print(f"Calculating spherical harmonics with degree {l_degree}...")

    # This function will calculate SPHARM coefficients and save them to CSV
    analysis_SHcPCA_One_embryo(
        embryo_path=embryo_path,
        used_degree=9,  # Used for PCA reduction
        l_degree=l_degree,  # Maximum spherical harmonic degree
        is_show_PCA=False,  # Set to True to visualize PCA components
    )

    print("Spherical harmonics calculation completed.")


def cluster_cells_by_shape(embryo_path, used_degree=9, cluster_num=12):
    """Cluster cells based on their shape features."""
    print(f"Clustering cells into {cluster_num} groups...")

    # This function performs K-means clustering on the shape features
    from threeDCSQ.analysis.SH_analyses import analysis_SHc_Kmeans_One_embryo

    analysis_SHc_Kmeans_One_embryo(
        embryo_path=embryo_path,
        used_degree=used_degree,  # Degree used for analysis
        cluster_num=cluster_num,  # Number of clusters
        is_normalization=True,  # Whether to normalize features
        is_show_cluster=False,  # Set to True to visualize clusters
    )

    print("Cell clustering completed.")


def analyze_specific_cell(embryo_path, frame_num, cell_label):
    """
    分析特定帧中特定标签的细胞形状特征。

    参数:
        embryo_path (str): 胚胎数据路径
        frame_num (str): 帧号，例如 '001'
        cell_label (int): 细胞标签编号

    返回:
        tuple: 包含 (cell_surface, center, sh_coefficient_instance, reconstruction)
    """
    import utils.cell_func as cell_f
    import utils.draw_func as draw_f
    from transformation.SH_represention import sample_and_SHc_with_surface
    from utils.sh_cooperation import do_reconstruction_from_SH
    import matplotlib.pyplot as plt
    from matplotlib import cm
    import nibabel as nib
    import numpy as np

    print(f"分析第 {frame_num} 帧中的细胞 {cell_label}...")

    # 加载特定帧的体积数据
    file_path = (
        f"{embryo_path}/{os.path.basename(embryo_path)}_{frame_num}_segCell.nii.gz"
    )
    volume = nib.load(file_path).get_fdata().astype(np.int16)

    # 提取细胞表面和中心点
    cell_surface, center = cell_f.nii_get_cell_surface(volume, cell_label)
    print(f"细胞表面点数量: {len(cell_surface)}")
    print(f"细胞中心位置: {center}")

    # 计算体积和表面积
    cell_volume = (volume == cell_label).sum()
    cell_surface_area = len(cell_surface) * 1.2031  # 表面积乘以权重
    irregularity = (cell_surface_area ** (1 / 2)) / (cell_volume ** (1 / 3))

    print(f"细胞体积: {cell_volume}")
    print(f"细胞表面积: {cell_surface_area}")
    print(f"不规则度 (表面积^(1/2) / 体积^(1/3)): {irregularity}")

    # 计算球谐系数
    l_degree = 16  # 球谐系数的最大度数
    sh_coefficient_instance = sample_and_SHc_with_surface(
        surface_points=cell_surface,
        sample_N=50,  # 表面采样点数
        lmax=l_degree,  # 最大度数
        surface_average_num=5,  # 每个采样点的平均点数
    )

    # 从球谐系数重建细胞形状
    reconstruction = do_reconstruction_from_SH(30, sh_coefficient_instance)

    # 可视化细胞形状
    fig = plt.figure(figsize=(12, 5))

    # 原始表面点
    ax1 = fig.add_subplot(121, projection="3d")
    draw_f.draw_3D_points(
        cell_surface - center, fig_name="原始表面", ax=ax1, cmap="viridis"
    )

    # 重建表面
    ax2 = fig.add_subplot(122, projection="3d")
    draw_f.draw_3D_points(
        reconstruction,
        fig_name=f"球谐重建 (度数: {l_degree})",
        ax=ax2,
        cmap="plasma",
    )

    plt.tight_layout()

    # 保存图像
    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/cell_{cell_label}_frame_{frame_num}_analysis.png", dpi=300)
    plt.close()

    # 保存球谐系数到CSV
    import pandas as pd
    from utils.sh_cooperation import flatten_clim, get_flatten_ldegree_morder

    coeffs_flat = flatten_clim(sh_coefficient_instance.coeffs)
    coeffs_df = pd.DataFrame(
        [coeffs_flat],
        columns=get_flatten_ldegree_morder(l_degree),
        index=[f"{frame_num}::{cell_label}"],
    )

    os.makedirs("results/coefficients", exist_ok=True)
    coeffs_df.to_csv(
        f"results/coefficients/cell_{cell_label}_frame_{frame_num}_coeffs.csv"
    )

    print(f"细胞 {cell_label} 分析完成。可视化结果和系数已保存到 'results' 文件夹")

    return cell_surface, center, sh_coefficient_instance, reconstruction


def main():
    # Directory containing the NIfTI files
    data_dir = "data"

    # Step 1: Load and process the embryo
    embryo_path = load_and_process_embryo(data_dir)

    # Step 2: Calculate volume and surface areas
    calculate_volume_surface(embryo_path)

    # Step 3: Calculate spherical harmonic coefficients using parallel processing
    get_SH_coefficient_of_embryo_parallel_npy(
        embryos_path_root=embryo_path,
        saving_path_root="DATA/spharm",
        sample_N=30,
        lmax=14,
        name_dictionary_path="DATA/name_dictionary.csv",
    )

    print("All analyses completed. Results are saved in the DATA directory.")


# Helper and parallel function for SH coefficient calculation and npy saving
def process_single_frame_sh(args):
    """Process all cells in a single frame."""
    (
        niigz_path,
        sample_N,
        surface_average_num,
        lmax,
        number_cell_affine_table,
        saving_path_root,
    ) = args

    try:
        # Extract timepoint info
        filename = os.path.basename(niigz_path)
        parts = filename.split("_")
        if len(parts) < 3:
            return None

        embryo_name = parts[0]
        tp_str = parts[-2]
        # Load and process embryo data
        embryo_array = general_f.load_nitf2_img(niigz_path).get_fdata().astype(int)
        cell_keys = np.unique(embryo_array)
        cell_keys = [k for k in cell_keys if k != 0]  # Remove background label

        # Process all cells in this frame
        results = []
        for label in cell_keys:
            try:
                # Get cell name from label
                name = number_cell_affine_table.get(label, f"cell_{label}")

                # Create cell directory if it doesn't exist
                cell_dir = os.path.join(saving_path_root, name)
                os.makedirs(cell_dir, exist_ok=True)

                cell_surface, center = cell_f.nii_get_cell_surface(embryo_array, label)
                points_membrane_local = cell_surface - center

                from threeDCSQ.transformation.SH_represention import do_sampling_with_interval
                from threeDCSQ.utils import sh_cooperation
                import pyshtools as pysh

                griddata, _ = do_sampling_with_interval(
                    sample_N, points_membrane_local, surface_average_num
                )
                sh_coefficient = pysh.expand.SHExpandDH(griddata, sampling=2, lmax_calc=lmax)
                cilm = sh_cooperation.flatten_clim(sh_coefficient)

                # Save cell data in its own directory
                cell_path = os.path.join(cell_dir, f"{name}_{tp_str}_l{lmax+1}.npy")
                np.save(cell_path, np.array(cilm))

                results.append((name, tp_str))
            except Exception as e:
                print(f"Error processing cell {label} ({name}) in frame {tp_str}: {str(e)}")
                continue

        return tp_str, len(results)
    except Exception as e:
        print(f"Error processing frame {niigz_path}: {str(e)}")
        return None


def get_SH_coefficient_of_embryo_parallel_npy(
    embryos_path_root,
    saving_path_root,
    sample_N,
    lmax,
    name_dictionary_path,
    surface_average_num=3,
):
    """Calculate SH coefficients for all cells in an embryo using parallel processing."""
    import pandas as pd
    from multiprocessing import Pool
    import glob
    import os
    import numpy as np
    from threeDCSQ.transformation.SH_represention import get_flatten_ldegree_morder
    from threeDCSQ.utils import sh_cooperation
    from tqdm import tqdm

    print(f"\nStarting SH coefficient calculation...")
    print(f"Input directory: {embryos_path_root}")
    print(f"Output directory: {saving_path_root}")
    print(f"Spherical harmonic degree: {lmax}")
    print(f"Sample points per cell: {sample_N}")
    
    # Get cell name mapping
    number_cell_affine_table, _ = cell_f.get_cell_name_affine_table(
        path=name_dictionary_path
    )
    
    # Get all nii.gz files
    niigz_files_this = sorted(glob.glob(os.path.join(embryos_path_root, "*.nii.gz")))
    print(f"\nFound {len(niigz_files_this)} frames to process")

    # Create saving directory if it doesn't exist
    os.makedirs(saving_path_root, exist_ok=True)
    
    # Prepare arguments for parallel processing
    args_list = [
        (
            niigz_path,
            sample_N,
            surface_average_num,
            lmax,
            number_cell_affine_table,
            saving_path_root,
        )
        for niigz_path in niigz_files_this
    ]

    # Process frames in parallel with progress bar
    num_cpus = min(48, mp.cpu_count())
    print(f"\nUsing {num_cpus} CPU cores for parallel processing")
    
    with mp.Pool(num_cpus) as pool:
        results = list(tqdm(
            pool.imap(process_single_frame_sh, args_list),
            total=len(niigz_files_this),
            desc="Processing frames"
        ))

    # Print summary
    successful_frames = [r for r in results if r is not None]
    print(f"\nProcessing completed:")
    print(f"Successfully processed {len(successful_frames)} frames out of {len(niigz_files_this)}")
    total_cells = sum(cell_count for _, cell_count in successful_frames)
    print(f"Total cells processed: {total_cells}")


if __name__ == "__main__":
    get_SH_coefficient_of_embryo_parallel_npy(
        embryos_path_root="DATA/SegmentCellUnified/WT_Sample1LabelUnified",
        saving_path_root="DATA/spharm",
        sample_N=30,
        lmax=14,
        name_dictionary_path=r"DATA/name_dictionary.csv",
    )
