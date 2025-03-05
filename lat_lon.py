import numpy as np
import pandas as pd

def generate_grid(lat_min, lat_max, lon_min, lon_max, grid_size=128):
    # 计算原始纬度和经度范围
    lat_range = lat_max - lat_min
    lon_range = lon_max - lon_min

    # 计算新的经纬度范围，使得每个格网的大小一致
    lat_step = lat_range / grid_size
    lon_step = lon_range / grid_size

    # 扩展经纬度范围以确保每个格网的大小一致
    new_lat_range = lat_step * grid_size
    new_lon_range = lon_step * grid_size

    # 更新纬度和经度的最大值
    new_lat_max = lat_min + new_lat_range
    new_lon_max = lon_min + new_lon_range

    # 创建格网
    grid_data = []
    grid_id = 0
    for i in range(grid_size):
        for j in range(grid_size):
            lat_start = lat_min + i * lat_step
            lat_end = lat_min + (i + 1) * lat_step if i + 1 < grid_size else new_lat_max
            lon_start = lon_min + j * lon_step
            lon_end = lon_min + (j + 1) * lon_step if j + 1 < grid_size else new_lon_max
            grid_data.append([grid_id, lat_start, lat_end, lon_start, lon_end])
            grid_id += 1

    return grid_data, new_lat_max, new_lon_max

def save_grid_to_csv(grid_data, filename):
    df = pd.DataFrame(grid_data, columns=["Grid_ID", "Lat_Min", "Lat_Max", "Lon_Min", "Lon_Max"])
    df.to_csv(filename, index=False,encoding='utf-8-sig')

def latlon(input_file,output_file,size):
    # 读取原始数据
    # file_path='data/wh_data_cleaned.csv'
    file_path=input_file
    # file_path='result/256/before/before_2020_02_12.csv'
    # file_path='result/256/after/after_2020_02_12.csv'

    column_to_read=['created_at','content','lon','lat']
    reviews = pd.read_csv(file_path,usecols=column_to_read)

    # 获取经纬度范围
    lat = reviews['lat']
    lon = reviews['lon']
    lat_min = min(lat)# 纬度最小值
    lat_max = max(lat)# 纬度最大值
    lon_min = min(lon)# 经度最小值
    lon_max = max(lon) # 经度最大值
    # lat_min = 29.94976043701172# 纬度最小值
    # lat_max = 31.365966796875# 纬度最大值
    # lon_min =113.69476318359375# 经度最小值
    # lon_max = 115.11096954345703 # 经度最大值
    # file='result/256/before'
    file=output_file

    grid_data, new_lat_max, new_lon_max = generate_grid(lat_min, lat_max, lon_min, lon_max, grid_size=size)

    # 输出格网数据到CSV文件
    save_grid_to_csv(grid_data, f"{file}/grid_lat_lon.csv")

    print(f"Grid has been divided into 128x128 grids, with the new lat_max: {new_lat_max} and lon_max: {new_lon_max}.")
