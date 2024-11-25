import datetime
import numpy as np
from pandasql import sqldf
import pandas as pd
from pandas import DataFrame

'''
input
'''
table_index = 3

'''频繁项分析代码'''
alarm_l1_total = dict()
alarm_l2_total = dict()

def update_event(alarm_l1_dict, alarm_l2_dict):
    confid_dict = {}
    for _tuple, count in alarm_l2_dict.items():
        #print(count, _tuple[0], alarm_l1_dict.get(_tuple[0]), _tuple[1], alarm_l1_dict.get(_tuple[1]))
        try:
            confidence0 = count / alarm_l1_dict.get(_tuple[0])
        except ZeroDivisionError as e:
            raise ValueError from e
        confid_dict[(_tuple[0], _tuple[1])] = confidence0
        try:
            confidence1 = count / alarm_l1_dict.get(_tuple[1])
        except ZeroDivisionError as e:
            raise ValueError from e
        confid_dict[(_tuple[1], _tuple[0])] = confidence1
    return confid_dict

def scan_batch(df: DataFrame, window: int, step: int) -> tuple:
    '''
    alarm_1_total: 每个类型告警出现的次数
    alarm_2_total: 每两个类型告警组合出现的次数
    '''
    df.index = df['incident_open_time']
    alarm_l1_set = set(list(zip(df['ci_type'],
                                df['alarm_type'],
                                df['alarm_type_group'])))
    alarm_l1_dict = {i: 0 for i in alarm_l1_set}

    alarm_l2_dict = {}
    start_time = df.index[0]
    start_time = datetime.datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S.%f")
    end_time = start_time + datetime.timedelta(seconds=window)
    window_data = df[str(start_time):str(end_time)]
    while start_time + pd.Timedelta(minutes=30) < datetime.datetime.strptime(df.index[-1], "%Y-%m-%d %H:%M:%S.%f"):
        if len(window_data) >= 2:
            window_alarm_set = set(list(zip(window_data['ci_type'],
                                            window_data['alarm_type'],
                                            window_data['alarm_type_group'])))
            window_alarm_dict = {
                i: alarm_l1_dict[i] + 1
                for i in window_alarm_set
            }
            alarm_l1_dict.update(window_alarm_dict)
            candidate = list(window_alarm_set)
            for i, value1 in enumerate(candidate):
                for _, value2 in enumerate(candidate[i + 1:]):
                    if value1[-1] == value2[-1]:
                        key_tmp = (value1, value2)
                        alarm_l2_dict[key_tmp] = \
                            alarm_l2_dict.get(key_tmp, 0) + 1

        start_time = start_time + pd.Timedelta(seconds=step)
        end_time = start_time + pd.Timedelta(seconds=(window + step))
        window_data = df[str(start_time): str(end_time)]

    alarm_l1_batch = {
        i: alarm_l1_total.get(i, 0) + alarm_l1_dict.get(i)
        for i in alarm_l1_dict
    }
    alarm_l1_total.update(alarm_l1_batch)

    alarm_l2_batch = {
        i: alarm_l2_total.get(i, 0) + alarm_l2_dict.get(i)
        for i in alarm_l2_dict
    }
    alarm_l2_total.update(alarm_l2_batch)
    return alarm_l1_total, alarm_l2_total

'''window 计算代码'''
def window_calculator(df: DataFrame):
    window = abs(datetime.datetime.strptime(df['incident_open_time'].loc[901], "%Y-%m-%d %H:%M:%S.%f") -
                 datetime.datetime.strptime(df['incident_open_time'].loc[0], "%Y-%m-%d %H:%M:%S.%f"))
    window = round(window.days * 1440 + window.seconds/60)
    step = round(window/3)
    return window, step

'''window时间内记录个数'''
def counter(i, window, df2):
    count = 0
    start_time = datetime.datetime.strptime(df2['incident_open_time'].loc[i], "%Y-%m-%d %H:%M:%S.%f")
    end_time = start_time - datetime.timedelta(seconds=window)
    for j in range(i, len(df2['incident_open_time'])):
        time = datetime.datetime.strptime(df2['incident_open_time'].loc[j], "%Y-%m-%d %H:%M:%S.%f")
        if time > end_time:
            count = count + 1
    return count - 1

'''main func'''
table_list = ['compressed_eAPPOps.xlsx',
              'compressed_eOMC.xlsx',
              'compressed_eSight_AF.xlsx',
              'compressed_HCS.xlsx']
window_list = [91,
               1421,
               1385,
               8100]
save_path = 'D:/PythonProject0707/'
k_list = ['cause-1', 'cause-1-score',
          'cause-2', 'cause-2-score',
          'cause-3', 'cause-3-score',
          'cause-4', 'cause-4-score',
          'cause-5', 'cause-5-score',]
table_name = table_list[table_index]
table = pd.read_excel(table_name)
df = pd.DataFrame(table)
df2 = pd.DataFrame(table)
size = len(df['index'])

sql1 = 'select * from df order by incident_open_time;'
df = sqldf(sql1, locals())
# print(sqldf('select incident_open_time from df limit 10', locals())

window = window_list[table_index]
step = round(window/3)

alarm_l1_total, alarm_l2_total = scan_batch(df, window, step)

confid_dict = update_event(alarm_l1_total, alarm_l2_total)


cause_list = [[0 for j in range(10)] for i in range(size)]

for i in range(0 ,size):
    x = 0
    y = x + 1
    max = i + 1 + 5
    if max > size:
        max = size
    for j in range(i + 1, max):
        #print(df2['index'].loc[i], df2['index'].loc[j])
        if df2['alarm_type'].loc[i] == df2['alarm_type'].loc[j]:
            continue

        alarm_1 = [df2['ci_type'].loc[i], df2['alarm_type'].loc[i], df2['alarm_type_group'].loc[i]]
        alarm_1 = tuple(alarm_1)
        alarm_2 = [df2['ci_type'].loc[j], df2['alarm_type'].loc[j], df2['alarm_type_group'].loc[j]]
        alarm_2 = tuple(alarm_2)
        try:
            cause_list[i][x] = df2['_id'].loc[j]
            cause_list[i][y] = round(confid_dict[(alarm_2, alarm_1)]*100, 4)
            x += 2
            y = x + 1
        except KeyError:
            pass

for i in range(0, size):
    for j in [1, 3, 5, 7, 9]:
        if cause_list[i][j] == 0:
            cause_list[i][j - 1] = 0

df_output = pd.DataFrame(data=cause_list, columns=k_list)
df_output = df_output.replace(0, np.nan)
df_output.to_csv(save_path + table_name + '_casue.csv', encoding='gbk', mode='w')