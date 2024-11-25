import pandas as pd
import numpy as np
import os
import sys
sys.path.append('..')


def extract_time_points(df):
    begin_time_points = df['incident_open_timestamp']
    end_time_points = df['incident_close_timestamp']
    sorted_time_steps = pd.concat([begin_time_points, end_time_points]).sort_values().drop_duplicates(
        keep='first').reset_index()[[0]].rename(columns={0: 'timestamp'})
    time_steps2index = sorted_time_steps.set_index('timestamp')
    time_steps2index['index'] = np.arange(len(time_steps2index))
    return sorted_time_steps, time_steps2index


def extract_time_intervals(ts):
    intervals = pd.concat([ts['timestamp'][:-1].reset_index().rename(columns={'timestamp': 'begin'}), ts['timestamp'][1:].reset_index().rename(columns={'timestamp': 'end'})], axis=1)[['begin', 'end']].reset_index()
    return intervals


def alarm_snapshot_iterator(df,  split_length=1000):

    id2ts, ts2id = extract_time_points(df)
    time_intervals = extract_time_intervals(id2ts)

    time_points = []
    df = df.reset_index()
    alarm_snapshots = {}
    for i, item in df.iterrows():
        time_points.append(
            (ts2id.loc[item['incident_open_timestamp']]['index'], i, True)
        )
        time_points.append(
            (ts2id.loc[item['incident_close_timestamp']]['index'], i, False)
        )

    time_points = sorted(time_points)

    existed_alarms = set()

    if 'alarm_group_id' not in df:
        df['alarm_group_id'] = 0

    alarm_group_id_set = df['alarm_group_id'].drop_duplicates().to_list()

    added = False
    size_sum = 0
    # ci_graph = CIGraph()
    for i, (timestamp_index, ind, is_add) in enumerate(time_points):
        if is_add:
            existed_alarms.add(ind)
            added = True
        else:
            existed_alarms.remove(ind)
        # if timestamp_index < 45617:
        #     continue

        # pdb.set_trace()
        # if i < len(time_points) - 1 and timestamp_index != time_points[i+1][0] \
        #         and timestamp_index < len(time_intervals) and added:

        if i < len(time_points) - 1 and timestamp_index != time_points[i+1][0] \
                and timestamp_index < len(time_intervals):
            # if len(existed_alarms) <= 1:
            #     continue
            # 只有跟上次生成的df相比有新增报警时，才生成新的df
            added = False
            # if len(existed_alarms) == 0:
            #     merged_df = pd.DataFrame([
            #         {'alarm_type': '无告警', '_id': None, 'ci_type_ci': None,
            #          'ci_name': None, 'ci_type': None}
            #     ])
            # else:

            merged_df = df[
                ['alarm_type', 'alarm_description', '_id', 'ci_id', 'ci_name', 'ci_type',
                 'alarm_group_id', 'incident_open_timestamp', 'incident_close_timestamp']
            ].iloc[list(existed_alarms)]

            # 对于没有出现告警的情况，需要补上无告警切片，以保证各个报警类型的边缘概率分布是正确的
            no_alarm_record = []
            for alarm_group_id in alarm_group_id_set:
                if len(merged_df.loc[merged_df['alarm_group_id'] == alarm_group_id]) == 0:
                    no_alarm_record.append({
                        'alarm_type': '无告警', '_id': None, 'ci_type_ci': None,
                        'ci_name': None, 'ci_type': None, 'incident_open_timestamp': None,
                        'incident_close_timestamp': None, 'alarm_group_id': alarm_group_id, 'alarm_description': None
                    })
            if len(no_alarm_record) != 0:
                merged_df = pd.concat([merged_df, pd.DataFrame(no_alarm_record)])

            merged_df['belong'] = timestamp_index
            merged_df['open_timestamp'] = time_intervals.iloc[timestamp_index]['begin']
            merged_df['close_timestamp'] = time_intervals.iloc[timestamp_index]['end']
            # merged_df = split_alarms_by_ci_connection(merged_df, union_set)

            alarm_snapshots[timestamp_index] = merged_df
            size_sum += len(merged_df)

        if len(alarm_snapshots.keys()) != 0 and len(alarm_snapshots.keys()) % 100 == 0:
            print('Timestamp_index: {}, Averaged size: {:.2f}'.format(
                timestamp_index, size_sum/len(alarm_snapshots.keys())
            ))
        if len(alarm_snapshots.keys()) >= split_length:
            yield alarm_snapshots, pd.concat([v for _, v in alarm_snapshots.items()])
            alarm_snapshots = {}
            size_sum = 0


def get_alarm_snapshot(df, return_all=False, split_length=1000):
    if return_all:
        xs = [x for _, x in alarm_snapshot_iterator(df, split_length=split_length)]
        return pd.concat(xs)
    else:
        for x in alarm_snapshot_iterator(df, split_length=split_length):
            yield x

