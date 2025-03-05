import pandas as pd
import os

def comment(file,emotion_file):

    # 读取细分格网和原始格网数据
    data = pd.read_csv(f'{file}/grid_lat_lon.csv')  # 细化后的格网

    # 读取评论数据（假设包含 'longitude' 和 'latitude' 列）
    comments_data = pd.read_csv(emotion_file)  # 评论数据文件

    # 确保格网数据已按经纬度列进行排序，以加速查找
    data_sorted = data.sort_values(by=['Lon_Min', 'Lat_Min'])

    # 使用一个函数批量处理经纬度区间匹配
    def get_grid_id(lon, lat, grid_data):
        # 通过查找经纬度区间来获取 Grid_ID
        match = grid_data[(grid_data['Lon_Min'] <= lon) & (grid_data['Lon_Max'] > lon) &
                           (grid_data['Lat_Min'] <= lat) & (grid_data['Lat_Max'] > lat)]
        return match['Grid_ID'].iloc[0] if not match.empty else None

    # 将格网数据转为 DataFrame 索引以加速查找
    grid_index = pd.MultiIndex.from_frame(data[['Lon_Min', 'Lon_Max', 'Lat_Min', 'Lat_Max']], names=['Lon_Min', 'Lon_Max', 'Lat_Min', 'Lat_Max'])

    # 为评论数据添加 Grid_ID 列（细分格网）
    comments_data['Grid_ID'] = comments_data.apply(
        lambda row: get_grid_id(row['lon'], row['lat'], data_sorted), axis=1
    )

    # 删除 Grid_ID 列为空的行
    comments_data = comments_data.dropna(subset=['Grid_ID'])

    # 创建 DataFrame
    df_xi = pd.DataFrame(comments_data)

    # 按 'Grid_ID' 进行分组
    grouped_xi = df_xi.groupby('Grid_ID')

    # 判断文件夹是否存在
    output = f'{file}/result'
    if not os.path.exists(output):
        os.makedirs(output)

    # 遍历每个组，将其保存为单独的 CSV 文件
    for group_name, group_data in grouped_xi:
        output_file = f'{output}/output_{group_name}.csv'
        group_data.to_csv(output_file, index=False, encoding='utf-8-sig')

    print("原始格网内的评论已保存为 'comments.csv'")

# file = 'result/256/before'
# file='result/sh/128'
# comment(file,'data/sh_output_emo.csv')