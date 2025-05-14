"""
可视化模块，用于显示NIfTI文件中的细胞边界和标注。
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # 设置后端为 TkAgg
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from typing import Union, Optional, Dict
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import os
import json
import colorsys
from ..data.reader import load_nifti_file, get_all_nifti_files, process_nifti_files
from ..utils.feature_utils import get_cell_labels
from ..utils.file_utils import get_cell_id_from_label


class CellBoundaryVisualizer:
    def __init__(self, data_dir: Union[str, Path]):
        """
        初始化可视化器。

        Args:
            data_dir: 包含NIfTI文件的目录路径
        """
        self.data_dir = Path(data_dir)
        self.volumes = {}
        self.current_frame = 0
        self.fig = None
        self.ax = None
        self.slider_frame = None
        self.rotation_angle = 0
        self.cell_labels: Dict[int, str] = {}
        self.cache_dir = Path("data_npy")
        self.boundaries_cache = {}
        self.centers_cache = {}
        self.cell_colors = {}

    def precompute_all_centers(self):
        """预计算所有帧的细胞中心点，如果已存在缓存则跳过。"""
        print("预计算所有帧的细胞中心点...")
        for frame_key, volume in tqdm(self.volumes.items(), desc="计算中心点"):
            cache_path = self.cache_dir / f"centers_{frame_key}.json"
            if cache_path.exists():
                # 已有缓存，跳过
                continue
            centers = self.get_cell_centers(volume)
            centers_dict = {str(int(k)): v.tolist() for k, v in centers.items()}
            with open(cache_path, 'w') as f:
                json.dump(centers_dict, f)
            self.centers_cache[frame_key] = centers

    def save_video(self, output_path: Union[str, Path], dpi: int = 100):
        """
        保存不旋转的视频帧

        Args:
            output_path: 输出目录路径
            dpi: 图像DPI
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("正在保存视频帧...")
        # 创建临时图形对象
        temp_fig = plt.figure(figsize=(12, 10))
        temp_ax = temp_fig.add_subplot(111, projection='3d')
        
        # 设置固定的视角
        temp_ax.view_init(elev=20, azim=0)
        
        for frame_idx in tqdm(range(len(self.volumes)), desc="保存帧"):
            frame_key = list(self.volumes.keys())[frame_idx]
            volume = self.volumes[frame_key]
            
            # 获取边界和中心点
            boundaries = self._get_cached_boundaries(frame_key, volume)
            centers = self._get_cached_centers(frame_key, volume)
            
            # 生成颜色（如果还没有）
            if not self.cell_colors:
                labels = get_cell_labels(volume)
                self.cell_colors = self._generate_cell_colors(labels)
            
            temp_ax.clear()
            
            # 绘制细胞
            for label in self.cell_colors.keys():
                mask = (volume == label)
                x, y, z = np.where(mask)
                if len(x) > 0:
                    color = self.cell_colors[label]
                    temp_ax.scatter(x, y, z, c=[color], alpha=0.3, s=1)
            
            # 添加细胞标注
            for label, center in centers.items():
                cell_id = get_cell_id_from_label(label)
                color = self.cell_colors[label]
                temp_ax.text(center[0], center[1], center[2], cell_id, 
                           color=color, fontsize=8)
            
            temp_ax.set_title(f"Frame {frame_key}")
            
            # 保存帧
            frame_path = output_path / f"frame_{int(frame_key):03d}.png"
            temp_fig.savefig(frame_path, dpi=dpi, bbox_inches='tight')
        
        plt.close(temp_fig)
        print(f"视频帧已保存到: {output_path}")
        print("您可以使用以下命令将帧转换为视频：")
        print(f"ffmpeg -framerate 10 -i {output_path}/frame_%03d.png -c:v libx264 -pix_fmt yuv420p output.mp4")

    def _generate_cell_colors(self, labels: np.ndarray) -> Dict[int, np.ndarray]:
        """
        为每个细胞生成独特的颜色

        Args:
            labels: 细胞标签数组

        Returns:
            Dict[int, np.ndarray]: 标签到颜色的映射
        """
        colors = {}
        n_labels = len(labels)
        
        # 使用黄金分割比来生成均匀分布的颜色
        golden_ratio = 0.618033988749895
        
        for i, label in enumerate(labels):
            # 使用HSV颜色空间，固定饱和度和亮度
            h = (i * golden_ratio) % 1.0
            s = 0.8  # 饱和度
            v = 0.9  # 亮度
            
            # 转换为RGB
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            colors[label] = np.array([r, g, b])
        
        return colors

    def _load_single_file(self, file_path: Path) -> tuple:
        """
        加载单个NIfTI文件，优先使用缓存

        Args:
            file_path: NIfTI文件路径

        Returns:
            tuple: (frame_num, volume_data)
        """
        frame_num = file_path.stem.split("_")[-2]
        cache_path = self.cache_dir / f"frame_{frame_num}.npy"
        
        if cache_path.exists():
            # 使用内存映射加载缓存文件
            volume = np.load(cache_path, mmap_mode='r')
        else:
            # 如果没有缓存，加载原始文件并创建缓存
            volume = load_nifti_file(str(file_path))
            self.cache_dir.mkdir(exist_ok=True)
            np.save(cache_path, volume)
        
        return frame_num, volume

    def _get_cached_boundaries(self, frame_key: str, volume: np.ndarray) -> np.ndarray:
        """
        获取缓存的边界信息

        Args:
            frame_key: 帧标识
            volume: 体积数据

        Returns:
            np.ndarray: 边界数据
        """
        cache_path = self.cache_dir / f"boundaries_{frame_key}.npy"
        
        if cache_path.exists():
            return np.load(cache_path)
        else:
            boundaries = self.find_cell_boundaries(volume)
            np.save(cache_path, boundaries)
            return boundaries

    def _get_cached_centers(self, frame_key: str, volume: np.ndarray) -> Dict[int, np.ndarray]:
        """
        获取缓存的细胞中心点信息

        Args:
            frame_key: 帧标识
            volume: 体积数据

        Returns:
            Dict[int, np.ndarray]: 细胞中心点字典
        """
        cache_path = self.cache_dir / f"centers_{frame_key}.json"
        
        if cache_path.exists():
            with open(cache_path, 'r') as f:
                centers_dict = json.load(f)
            # 将列表转换回numpy数组，并兼容 '1.0' 这种 key
            return {int(float(k)): np.array(v) for k, v in centers_dict.items()}
        else:
            centers = self.get_cell_centers(volume)
            # 将numpy数组转换为列表以便JSON序列化，key 强制为 int
            centers_dict = {str(int(k)): v.tolist() for k, v in centers.items()}
            with open(cache_path, 'w') as f:
                json.dump(centers_dict, f)
            return centers

    def load_data(self, n_workers: Optional[int] = None, force_reload: bool = False):
        """
        并行加载所有NIfTI文件数据，使用缓存机制

        Args:
            n_workers: 并行工作进程数，默认为CPU核心数
            force_reload: 是否强制重新加载所有文件（忽略缓存）
        """
        print("正在加载数据...")
        
        if force_reload:
            print("强制重新加载所有文件...")
            self.volumes = process_nifti_files(self.data_dir)
            return

        nifti_files = get_all_nifti_files(self.data_dir)
        
        if n_workers is None:
            n_workers = min(40, mp.cpu_count())
        
        print(f"使用 {n_workers} 个进程并行加载...")
        with mp.Pool(n_workers) as pool:
            results = list(tqdm(
                pool.imap(self._load_single_file, nifti_files),
                total=len(nifti_files),
                desc="加载文件"
            ))
        
        # 将结果存入volumes字典
        self.volumes = dict(results)
        print(f"成功加载 {len(self.volumes)} 个时间帧")

    def find_cell_boundaries(self, volume: np.ndarray) -> np.ndarray:
        """
        使用拉普拉斯算子找到细胞边界。

        Args:
            volume: 3D体积数据

        Returns:
            边界图像
        """
        from scipy.ndimage import laplace
        return np.abs(laplace(volume)) > 0

    def get_cell_centers(self, volume: np.ndarray) -> Dict[int, np.ndarray]:
        """
        计算每个细胞的中心点。

        Args:
            volume: 3D体积数据

        Returns:
            细胞标签到中心点的映射字典
        """
        centers = {}
        labels = get_cell_labels(volume)
        for label in tqdm(labels, desc="计算细胞中心点", leave=False):
            label = int(label)  # 强制转为int，避免float label
            mask = (volume == label)
            indices = np.where(mask)
            center = np.mean(indices, axis=1)
            centers[label] = center
        return centers

    def update(self, val):
        """更新显示的图像"""
        frame_idx = int(self.slider_frame.val)
        frame_key = list(self.volumes.keys())[frame_idx]
        volume = self.volumes[frame_key]
        
        print(f"正在处理第 {frame_key} 帧...")
        print("获取细胞边界...")
        boundaries = self._get_cached_boundaries(frame_key, volume)
        
        print("获取细胞中心点...")
        centers = self._get_cached_centers(frame_key, volume)

        # 生成细胞颜色
        if not self.cell_colors:
            labels = get_cell_labels(volume)
            self.cell_colors = self._generate_cell_colors(labels)

        self.ax.clear()
        
        print("绘制3D视图...")
        # 为每个细胞绘制边界
        for label in self.cell_colors.keys():
            mask = (volume == label)
            x, y, z = np.where(mask)
            if len(x) > 0:  # 确保细胞存在
                color = self.cell_colors[label]
                self.ax.scatter(x, y, z, c=[color], alpha=0.3, s=1)

        # 添加细胞标注
        for label, center in centers.items():
            cell_id = get_cell_id_from_label(label)
            color = self.cell_colors[label]
            self.ax.text(center[0], center[1], center[2], cell_id, 
                        color=color, fontsize=8)

        self.ax.set_title(f"Frame {frame_key}")
        self.ax.view_init(elev=20, azim=self.rotation_angle)
        self.fig.canvas.draw_idle()
        print("更新完成")

    def rotate(self, event):
        """旋转视图"""
        self.rotation_angle = (self.rotation_angle + 10) % 360
        self.update(None)

    def _on_key(self, event):
        """
        处理键盘事件
        
        Args:
            event: 键盘事件对象
        """
        if event.key == 'left':
            # 后退一帧
            new_val = max(0, self.slider_frame.val - 1)
            self.slider_frame.set_val(new_val)
        elif event.key == 'right':
            # 前进一帧
            new_val = min(len(self.volumes) - 1, self.slider_frame.val + 1)
            self.slider_frame.set_val(new_val)

    def visualize(self):
        """创建交互式可视化界面"""
        if not self.volumes:
            self.load_data()

        print("初始化可视化界面...")
        self.fig = plt.figure(figsize=(12, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        plt.subplots_adjust(bottom=0.2)

        # 创建帧滑块
        ax_frame = plt.axes([0.25, 0.1, 0.65, 0.03])
        self.slider_frame = Slider(
            ax_frame, "Frame", 0, len(self.volumes) - 1, valinit=0, valstep=1
        )

        # 创建旋转按钮
        ax_rotate = plt.axes([0.8, 0.1, 0.1, 0.03])
        rotate_button = Button(ax_rotate, 'Rotate')

        # 注册更新函数
        self.slider_frame.on_changed(self.update)
        rotate_button.on_clicked(self.rotate)
        
        # 注册键盘事件
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        # 显示初始图像
        self.update(0)
        print("可视化界面准备就绪")
        print("使用说明：")
        print("- 左右方向键：前进/后退一帧")
        print("- 滑块：直接跳转到指定帧")
        print("- 旋转按钮：旋转视图")
        plt.show()


def main():
    """主函数"""
    visualizer = CellBoundaryVisualizer(
        "DATA/SegmentCellUnified/WT_Sample1LabelUnified"
    )
    visualizer.load_data()
    visualizer.precompute_all_centers()  # 预计算所有中心点
    visualizer.save_video("output_frames")  # 保存视频帧
    visualizer.visualize()  # 显示交互式界面


if __name__ == "__main__":
    main()
