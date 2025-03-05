import pandas as pd

def time(input_file,output_before,output_after):
    # 读取CSV文件
    df = pd.read_csv(input_file)

    # 转换时间格式
    df['created_at'] = pd.to_datetime(df['created_at'], format='%Y/%m/%d %H:%M')

    # 定义分割日期
    split_date = pd.to_datetime('2020/2/12', format='%Y/%m/%d')

    # 按照时间分割数据
    before_split = df[df['created_at'] < split_date]
    after_split = df[df['created_at'] >= split_date]

    # 保存为两个新的CSV文件
    before_split.to_csv(output_before, index=False,encoding='utf-8-sig')
    after_split.to_csv(output_after, index=False,encoding='utf-8-sig')

# t('emotion_prediction_wh.csv','result/bert/wh/128/before/before_2020_02_12_new.csv','result/bert/wh/128/after/after_2020_02_12_new.csv')
