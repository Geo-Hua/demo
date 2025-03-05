import pandas as pd
import re
# 定义一个函数来删除HTML标签
def remove_html_tags(text):
    clean_text = re.sub(r'<[^>]+>', '', text)  # 使用正则表达式删除HTML标签
    return clean_text


def  cd(input_file,out_file):
    # 读取CSV文件
    df = pd.read_csv(input_file)
    # 假设评论列名为 'comment'，根据实际列名修改
    df['content'] = df['content'].apply(remove_html_tags)
    # 判断是否有重复行，并去掉重复行
    df_cleaned = df.drop_duplicates()

    # 保存处理后的CSV文件
    df_cleaned.to_csv(out_file, index=False,encoding='utf-8-sig')

    print("重复行已去除，清理后的文件已保存为 'cleaned_file.csv'.")

cd('data/wh_data.csv','data/wh_data_cleaned.csv')
