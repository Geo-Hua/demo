import math
import pickle
import webbrowser
import requests
import numpy as np
import folium
from scipy.ndimage import zoom
import pandas as pd
from caculate import caculate_index
class QuadTreeNode:
    def __init__(self, val=None, topLeft=None, topRight=None, bottomLeft=None, bottomRight=None, x=None, y=None,
                 size=None):
        self.val = val  # 值（叶子节点时，表示矩阵中的值）
        self.topLeft = topLeft  # 左上子节点
        self.topRight = topRight  # 右上子节点
        self.bottomLeft = bottomLeft  # 左下子节点
        self.bottomRight = bottomRight  # 右下子节点
        self.x = x  # 节点的左上角x坐标
        self.y = y  # 节点的左上角y坐标
        self.size = size  # 节点的大小


def can_merge(grid, x, y, size):
    """判断一个子矩阵是否可以合并为单一值"""
    val = grid[x][y]
    for i in range(x, x + size):
        for j in range(y, y + size):
            if grid[i][j] != val:
                return False
    return True


def merge(grid, x, y, size, rectangles):
    """递归合并，逐步向上构建四叉树，并记录合并的区域"""
    if size == 1 or can_merge(grid, x, y, size):
        # 叶子节点或可以合并的节点
        rectangles.append((x, y, size, grid[x][y]))  # 记录合并区域
        return QuadTreeNode(val=grid[x][y], x=x, y=y, size=size)

    mid = size // 2
    topLeft = merge(grid, x, y, mid, rectangles)
    topRight = merge(grid, x, y + mid, mid, rectangles)
    bottomLeft = merge(grid, x + mid, y, mid, rectangles)
    bottomRight = merge(grid, x + mid, y + mid, mid, rectangles)

    # 记录合并区域
    rectangles.append((x, y, size, None))

    return QuadTreeNode(topLeft=topLeft, topRight=topRight,
                        bottomLeft=bottomLeft, bottomRight=bottomRight, x=x, y=y, size=size)


def interpolate_grid(grid, target_size):
    """用插值法调整grid大小，使其变为目标大小"""
    original_size = len(grid)
    zoom_factor = target_size / original_size
    interpolated_grid = zoom(grid, zoom_factor, order=1)  # 使用最近邻插值
    return interpolated_grid


def plot_grid_on_map(grid, rectangles, color_map, n, output_file, min_lat, max_lat, min_lon, max_lon):
    """将合并后的矩阵区域绘制在地图上，并保存为HTML文件"""
    # 创建地图对象，中心点设为经纬度范围的中心
    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2
    # 创建一个蓝色调的地图
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles='CartoDB.DarkMatter',  # 使用CartoDB的深色底图
        control_scale=True,  # 启用比例尺
        attr="Map tiles by CartoDB, under CC BY 3.0. Data by OpenStreetMap, under ODbL."  # 添加版权声明
    )
    # 武汉边界（确保图层在网格下方）
    wuhan_geojson = requests.get('https://geojson.cn/api/china/420000.json').json()
    folium.GeoJson(wuhan_geojson,
                   style_function=lambda x: {'color': 'white', 'weight': 1.5},
                   name="武汉边界").add_to(m)
    # 计算每个矩形区域的实际经纬度范围
    lat_step = (max_lat - min_lat) / n
    lon_step = (max_lon - min_lon) / n

    # 为每个矩阵区域添加矩形覆盖物
    for grid_id,rect in enumerate(rectangles):
        x, y, size, value = rect
        if value is not None:  # 如果区域有值，给它上色
            color = color_map[value % len(color_map)]  # 根据值映射颜色，取余确保颜色重复
            # color = color_map[value]  # 根据值映射颜色，取余确保颜色重复
        else:
            color = 'none'  # 否则表示无效区域，不填充颜色

        # 计算矩形区域的实际经纬度范围
        # lat1 = max_lat - y * lat_step
        # lat2 = max_lat - (y + size) * lat_step
        # lon1 = min_lon + x * lon_step
        # lon2 = min_lon + (x + size) * lon_step
        lat1 = min_lat + y * lat_step
        lat2 = min_lat + (y + size) * lat_step
        lon1 = min_lon + x * lon_step
        lon2 = min_lon + (x + size) * lon_step
        # 将矩形区域添加到地图
        folium.Rectangle(
            bounds=[(lat1, lon1), (lat2, lon2)],
            color='black',
            # color=color,
            fill=value,
            weight=0.5,  # 设置边框宽度
            fill_color=color,
            fill_opacity=0.5
        ).add_to(m)


    # 保存地图为HTML文件
    m.save(output_file)


def construct_quad_tree(grid):
    n = len(grid)
    rectangles = []
    quad_tree = merge(grid, 0, 0, n, rectangles)
    return quad_tree, rectangles

def qt(file):
    # file='result/256'
    # 加载数据
    with open(f'{file}/clustering_results.pkl', 'rb') as f:
        grid = pickle.load(f)

    # 定义目标尺寸（可以是任意2的幂次方，如8x8，16x16）
    target_size = int(math.sqrt(grid.size))

    # 对grid进行插值填充
    interpolated_grid = interpolate_grid(grid, target_size)

    # 定义固定的颜色列表

    color_map = [
        "#FF6666","#B34747","#FFB366"
    ]
    # 构建四叉树并获取每次合并的区域
    quad_tree, rectangles = construct_quad_tree(interpolated_grid)
    caculate_index(interpolated_grid,128)
    data = pd.read_csv(f'{file}/grid_lat_lon.csv')  # 细化后的格网
    # 定义经纬度范围
    # min_lat, max_lat = 34.0, 38.0
    # min_lon, max_lon = -118.0, -114.0
    min_lat = min(data['Lat_Max'])# 纬度最小值
    max_lat = max(data['Lat_Min'])# 纬度最大值
    min_lon = min(data['Lon_Max'])# 经度最小值
    max_lon = max(data['Lon_Min']) # 经度最大值
    # 绘制带边界的矩阵图并保存为HTML文件
    plot_grid_on_map(interpolated_grid, rectangles, color_map, target_size, f'{file}/grid.html', min_lat, max_lat, min_lon, max_lon)

# file='result/1'
file='result/bert/wh/128'
# qt(file)