from clear_duplicate import cd
from bert import judge_emotion
from seperate_time import time
from lat_lon import latlon
from comments import comment
from graph import g
from cluster import main1
from tree import qt
cd('data/sh_data.csv','data/sh_data_cleaned.csv')
# judge_emotion('data/sh_data_cleaned.csv','data/sh_output_emo.csv')
time('emotion_prediction_wh.csv','result/bert/wh/128/before/before_2020_02_12.csv','result/bert/wh/128/after/after_2020_02_12.csv')
file='result/bert/wh/128/before'
size=256
latlon('emotion_prediction_wh.csv',file,size)
comment(file,'emotion_prediction_wh.csv')
g(file)
main1(file)
qt(file)