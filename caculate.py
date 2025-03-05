import numpy as np
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from collections import defaultdict
from collections import Counter
from scipy.spatial.distance import pdist, squareform
from scipy.stats import zscore
# 假设 `grid` 是 128×128 矩阵，每个单元的值表示聚类后的类别编号，0 代表未聚类区域
# grid = np.random.randint(0, 5, (128, 128))  # 示例数据，替换为你的聚类结果
def caculate_index(grid,num):
    # 1. **计算聚类后的网格数目**
    # 统计每个聚类块的尺寸
    def find_clusters(grid):
        visited = np.zeros_like(grid, dtype=bool)
        clusters = []  # 记录所有格网的尺寸

        def bfs(x, y, label):
            """使用 BFS 找到整个格网区域的大小"""
            queue = [(x, y)]
            visited[x, y] = True
            cluster_cells = [(x, y)]  # 记录该聚类的所有单元格

            while queue:
                cx, cy = queue.pop(0)
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # 四邻域搜索
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
                        if not visited[nx, ny] and grid[nx, ny] == label:
                            visited[nx, ny] = True
                            queue.append((nx, ny))
                            cluster_cells.append((nx, ny))

            return cluster_cells

        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid[i, j] > 0 and not visited[i, j]:  # 忽略 0（未聚类区域）
                    cluster_size = len(bfs(i, j, grid[i, j]))
                    clusters.append(cluster_size)

        return clusters

    clusters = find_clusters(grid)

    # 计算最终的网格数目
    num_final_grids = len(clusters)
    print(f"聚类后格网数目: {num_final_grids}")

    # 2. **计算平均网格大小**
    # 格网平均大小可以用面积或者边长衡量
    avg_grid_area = np.mean(clusters)
    avg_grid_length = np.mean([np.sqrt(size) for size in clusters])

    print(f"聚类后格网平均面积: {avg_grid_area:.2f} 个单元")
    print(f"聚类后格网平均边长: {avg_grid_length:.2f} 单元")


    # 6. **计算 Moran’s I 空间自相关指数**
    def morans_I(grid):
        """计算 Moran’s I 评估空间自相关"""
        valid_cells = [(x, y, grid[x, y]) for x in range(grid.shape[0]) for y in range(grid.shape[1]) if grid[x, y] > 0]
        if len(valid_cells) < 2:
            return None

        df = pd.DataFrame(valid_cells, columns=["x", "y", "value"])
        X = df["value"].values
        W = squareform(pdist(df[["x", "y"]]))  # 计算欧几里得距离
        W = np.exp(-W)  # 归一化权重
        W /= W.sum()  # 归一化

        X_mean = np.mean(X)
        num = np.sum(W * ((X - X_mean)[:, None] * (X - X_mean)))
        den = np.sum((X - X_mean) ** 2)

        return num / den


    morans_I_value = morans_I(grid)
    print(f"\nMoran’s I 空间自相关指数: {morans_I_value:.4f}")
    # 4. **类别占比**
    # 2. **统计网格数目**
    unique_classes, class_counts = np.unique(grid, return_counts=True)
    class_dict = dict(zip(unique_classes, class_counts))
    num_grids = sum(class_counts)  # 非空格网的数量
    class_ratios = {k: v / num_grids for k, v in class_dict.items()}
    print("\n类别占比:")
    for cls, ratio in class_ratios.items():
        print(f"类别 {cls}: {ratio:.2%}")


    # 假设 `original_grid` 是原始未聚类的 128×128 网格
    # `grid` 是四叉树聚类后的网格，值为 0、1、2、3、4（类别编号）


    # 1. **计算聚类后的网格数目**
    def find_clusters(grid):
        visited = np.zeros_like(grid, dtype=bool)
        clusters = []  # 记录所有格网的尺寸

        def bfs(x, y, label):
            """使用 BFS 找到整个格网区域的大小"""
            queue = [(x, y)]
            visited[x, y] = True
            cluster_cells = [(x, y)]  # 记录该聚类的所有单元格

            while queue:
                cx, cy = queue.pop(0)
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # 四邻域搜索
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
                        if not visited[nx, ny] and grid[nx, ny] == label:
                            visited[nx, ny] = True
                            queue.append((nx, ny))
                            cluster_cells.append((nx, ny))

            return cluster_cells

        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid[i, j] > 0 and not visited[i, j]:  # 忽略 0（未聚类区域）
                    cluster_size = len(bfs(i, j, grid[i, j]))
                    clusters.append(cluster_size)

        return clusters

    clusters = find_clusters(grid)

    # **计算最终的网格数目**
    num_final_grids = len(clusters)

    # **计算平均网格大小**
    avg_grid_area = np.mean(clusters)
    avg_grid_length = np.mean([np.sqrt(size) for size in clusters])

    # 3. **计算网格合并比例**
    original_num_grids = num * num  # 初始网格数量
    merge_ratio = (original_num_grids - num_final_grids) / original_num_grids

    # 4. **计算网格密度**
    # 假设单位面积（区域总面积 = 128×128）计算密度
    grid_density = num_final_grids / (num * num)

    # 5. **热点分析**（高密度网格聚集区域）
    import matplotlib.font_manager as fm

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文
    plt.rcParams['axes.unicode_minus'] = False  # 负号正常显示

    # 统计不同大小的格网数量
    # 假设一个 128x128 的矩阵，随机填充 0~3 的类别


    # 统计最终不同尺度格网的数量
    final_grid_counts = defaultdict(int)

    # 记录已经合并的格网
    visited = np.zeros((num, num), dtype=bool)

    # 递归执行四叉树合并
    def quadtree(matrix, x, y, size):
        """ 递归进行四叉树合并，统计最终合并的格网数量 """
        # global visited

        # 如果当前格网已经被合并，跳过
        if visited[x:x + size, y:y + size].all():
            return True  # 表示这个区域已合并

        # 取当前 size×size 格网的内容
        sub_matrix = matrix[x:x + size, y:y + size]

        # 判断是否可以合并（整个区域的值相同）
        if np.all(sub_matrix == sub_matrix[0, 0]):
            final_grid_counts[size] += 1  # 统计该尺度的格网数量
            visited[x:x + size, y:y + size] = True  # 标记该区域已合并
            return True  # 合并成功
        else:
            # 不能合并，递归拆分 4 个子区域
            new_size = size // 2
            if new_size > 0:
                quadtree(matrix, x, y, new_size)  # 左上
                quadtree(matrix, x + new_size, y, new_size)  # 右上
                quadtree(matrix, x, y + new_size, new_size)  # 左下
                quadtree(matrix, x + new_size, y + new_size, new_size)  # 右下
            return False  # 不能合并

    # **按照从大到小的尺度进行四叉树合并**
    sizes = [128, 64, 32, 16, 8, 4, 2, 1]
    for size in sizes:
        for i in range(0, num, size):
            for j in range(0, num, size):
                if not visited[i, j]:  # 只有未被合并的区域才尝试合并
                    quadtree(grid, i, j, size)

    # **输出最终统计结果**
    total_count = sum(final_grid_counts.values())  # 计算所有最终格网的总数
    for size, count in sorted(final_grid_counts.items(), reverse=True):
        print(f"{size}×{size} 级别的最终单元格数量: {count}")

    print(f"\n最终总单元格数量: {total_count}")

    # **打印结果**
    print(f"聚类后网格数目: {num_final_grids}")
    print(f"聚类后网格平均面积: {avg_grid_area:.2f} 个单元")
    print(f"聚类后网格平均边长: {avg_grid_length:.2f} 单元")
    print(f"网格合并比例: {merge_ratio:.2%}")
    print(f"网格密度: {grid_density:.6f} （单位面积网格数）")

    # **热点区域分析**
    # print("\n热点分析（不同大小网格的分布）：")
    # for size, count in sorted(hotspot_analysis.items(), key=lambda x: x[0]):
    #     print(f"网格大小 {size} 单元: {count} 个")



