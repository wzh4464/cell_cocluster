import numpy as np

# 加载 npy 文件
data = np.load('data_npy/frame_001.npy')

# 打印基本信息
print("数据类型:", data.dtype)
print("数据形状:", data.shape)
print("数组维度:", data.ndim)
print("数组大小:", data.size)

# 打印数据的基本统计信息
print("\n基本统计信息:")
print("最小值:", np.min(data))
print("最大值:", np.max(data))
print("平均值:", np.mean(data))
print("标准差:", np.std(data))

# 如果数组不是太大，打印前几个元素
print("\n数据前几个元素:")
print(data.flatten()[:10])  # 只打印前10个元素 

# 查找并打印唯一值
unique_values = np.unique(data)
print("\n唯一值列表:")
print(f"共有 {len(unique_values)} 个唯一值")
print("所有唯一值:", unique_values) 