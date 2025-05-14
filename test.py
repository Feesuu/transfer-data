import numpy as np
import os

import numpy as np
from tqdm import tqdm
import os

def backpack_memmap_with_selection(number, weight, w, v):
    # 创建临时文件
    temp_file = "temp_knapsack_0.npy"
    if os.path.exists(temp_file):
        os.remove(temp_file)

    # 初始化 memmap
    result = np.memmap(temp_file, dtype=np.float16, mode="w+", shape=(number+1, weight+1))
    result[0, :] = 0  # 初始条件

    # 动态规划（向量化）
    for i in tqdm(range(1, number+1), total=number, desc="Dynamic calculating..."):
        prev_row = result[i-1, :]
        current_row = prev_row.copy()  # 默认继承上一行的值

        # 向量化计算可能更新的位置
        j_values = np.arange(1, weight+1)
        mask = j_values >= w[i-1]  # 找到满足 j >= w[i-1] 的位置
        update_indices = j_values[mask]

        # breakpoint()
        if len(update_indices) > 0:
            # 计算新值：prev_row[j - w[i-1]] + v[i-1]
            new_values = prev_row[update_indices - w[i-1]] + v[i-1]
            # 取最大值
            current_row[update_indices] = np.maximum(prev_row[update_indices], new_values)

        result[i, :] = current_row
        result.flush()

    # 回溯选择的物品（与原代码相同）
    selected_items = []
    j = weight
    for i in tqdm(range(number, 0, -1), total=number, desc="Choosing..."):
        if result[i, j] != result[i-1, j]:
            selected_items.append(i-1)
            j -= w[i-1]

    max_value = result[number, weight]
    del result
    os.remove(temp_file)
    return max_value, selected_items

# 示例输入
number = 5
weight = 10
# w = np.array([2, 3, 4, 5, 9])
# v = np.array([3, 4, 5, 8, 10])

w = [2, 3, 4, 5, 9]
v = [3, 4, 5, 8, 10]

max_val, selected = backpack_memmap_with_selection(number, weight, w, v)
print(f"最大价值: {max_val}")          
print(f"选择的物品索引: {selected}")