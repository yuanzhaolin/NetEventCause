from sklearn.model_selection import KFold
from utils import update_index_df
import shutil
import pandas as pd
import os
import numpy as np


# def delete_time_repeated(alarm_df:pd.DataFrame):
#     """
#     删除掉同时间点的告警事件
#
#     :param alarm_df:
#     :return:
#     """
#
#
#     # is_repeat = (
#     #         alarm_df[['index', 'incident_open_timestamp']].iloc[:-1].values ==
#     #         alarm_df[['index', 'incident_open_timestamp']].iloc[1:].values
#     # ).all(axis=1)
#     alarm_df = alarm_df.drop_duplicates('incident_open_timestamp')
#
#     is_repeat = (
#             alarm_df[['incident_open_timestamp']].iloc[:-1].values ==
#             alarm_df[['incident_open_timestamp']].iloc[1:].values
#     ).all(axis=1)
#     alarm_df['is_repeated'] = np.concatenate([[False], is_repeat])
#     no_repeat_df = alarm_df.loc[~alarm_df['is_repeated']]
#     return no_repeat_df

def double_point(pos, max_dis):
    res = [-1 for _ in range(len(pos))]
    j = 0
    for i in range(len(pos)):
        while j < len(pos) and pos[j] - pos[i] < max_dis:
            j += 1
        res[i] = j
    return res


def count_event_kinds(seqs):
    from collections import defaultdict
    s = set()
    c = defaultdict(int)
    for seq in seqs:
        for e in seq:
            s.add(int(e[1]))
            c[int(e[1])] += 1
    return len(s), c, max(s)


def sample_seqs(
        df,
        seq_num=None,
        min_len=50,
        max_len=100,
        max_hours=2,
        time_weight=False
):

    if time_weight:
        sampling_weight = df['incident_open_timestamp'][1:].values - df['incident_open_timestamp'][:-1].values
        sampling_weight = np.concatenate([[0], sampling_weight])
    else:
        sampling_weight = np.ones(len(df))

    chosen_options_length = max(len(df) - min_len + 1, 0)
    sampling_weight = sampling_weight[:chosen_options_length]

    right_farthest_pos = double_point(
        (df['incident_open_timestamp']/1e9/3600).to_list()[:chosen_options_length],
        max_hours
    )
    reserved_pos = [True if rp-i >= min_len else False for i, rp in enumerate(right_farthest_pos)][:chosen_options_length]

    # 第一个位置的触发时间难以设定，直接删除
    if len(reserved_pos) >= 1:
        reserved_pos[0] = False

    reserved_pos = np.arange(chosen_options_length)[reserved_pos]
    sampling_weight = sampling_weight[reserved_pos]
    sampling_weight = sampling_weight / np.sum(sampling_weight)

    if len(reserved_pos) == 0:
        # 如果reserved_pos为空，说明过
        chosen_df = df.copy()
        chosen_df['seq_group'] = 0
        return [df2events_seq(chosen_df, start_time=df.iloc[0]['incident_open_timestamp'])], [chosen_df]

    seqs = []
    chosen_df_list = []

    indexes = np.random.choice(reserved_pos, seq_num, p=sampling_weight)

    for ind in indexes:
        chosen_df = df.iloc[ind: min(ind+max_len, right_farthest_pos[ind])].copy()
        # if chosen_df['incident_open_timestamp'].max() - chosen_df['incident_open_timestamp'].min() \
        #         > max_hours * 3600 * 1e9:
        #     continue
        # start_time = chosen_df['incident_open_timestamp'].min() / 1e10
        start_time = df.iloc[ind-1]['incident_open_timestamp'] + np.random.rand() * (
                df.iloc[ind]['incident_open_timestamp'] - df.iloc[ind-1]['incident_open_timestamp']
        )
        # seqs.append([(np.clip(t-start_time, 1e-3, None) / 1e11, alarm_type_ind) for t, alarm_type_ind in
        #              chosen_df[['incident_open_timestamp', 'index']].values.tolist()])
        seqs.append(df2events_seq(chosen_df, start_time=start_time))
        chosen_df['seq_group'] = len(seqs)
        chosen_df_list.append(chosen_df)
    return seqs, chosen_df_list


def df2events_seq(chosen_df, start_time=None, alarm_type_col='index', time_col='t'):
    """

    :param chosen_df:
    :param start_time:
    :param alarm_type_col: chosen_df中标记告警类型的列
    :param time_scale: 现实中的1s对应数据集中的time_scale
    :return:
    """
    if start_time is None:
        start_time = chosen_df.iloc[0][time_col]
    seq = [
        (t-start_time, alarm_type_ind) for t, alarm_type_ind in chosen_df[[time_col, alarm_type_col]].values.tolist()
    ]
    return seq


def generate_event_alarm_datasets(
        alarm_df, seq_num=None, n_types=None,  min_len=20, max_len=100, max_hours=2, save_path=None, name='data', alarm_type_col='index',
        name_suffix=None,
):
    # alarm_df = delete_time_repeated(alarm_df)

    alarm_df = alarm_df.sort_values(['incident_open_timestamp', alarm_type_col])
    # 不同类型，相同时间点的告警也只保留一个
    alarm_df = alarm_df.drop_duplicates('incident_open_timestamp')

    if seq_num is None:
        seq_num = len(alarm_df)

    event_seqs, chosen_df_list = sample_seqs(alarm_df, seq_num=seq_num, min_len=min_len, max_len=max_len, max_hours=max_hours)

    if event_seqs is None:
        return None, None

    num, dic, max_id = count_event_kinds(event_seqs)
    print('%s: %d kinds in %d seq. The max id is %d' % (name, num, len(event_seqs), max_id))

    # print(pd.DataFrame([[k, v] for k, v in dic.items()], columns=['index', 'cnt']).sort_values('cnt'))
    if len(event_seqs) < 5:
        event_seqs = event_seqs + [event_seqs[-1]] * (5-len(event_seqs))
    train_test_splits = list(
        KFold(5, shuffle=True, random_state=42).split(
            range(len(event_seqs))
        )
    )
    output_dir = os.path.join(save_path, f"{name}")
    # if name_suffix:
    #     output_dir = output_dir + '_' + name_suffix

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'data.npz')

    np.savez_compressed(
        output_path,
        event_seqs=np.array(event_seqs, dtype='object'),
        train_test_splits=np.array(train_test_splits, dtype='object'),
        n_types=n_types,
    )

    # excel_path = os.path.join(output_dir, 'excels')

    # os.makedirs(excel_path, exist_ok=True)
    # for i, df in enumerate(chosen_df_list):
    #     assert isinstance(df, pd.DataFrame)
    #     df.to_excel(
    #         os.path.join(excel_path, '%s.xlsx' % str(i))
    #     )
            # str(i))

    print('Generated dataset is saved in %s' % output_dir)
    return np.load(output_path, allow_pickle=True), output_dir


def alarm_df2event_dataset(df, ignore_group_id=False, group_col='alarm_type_group', alarm_type_col='index',
                           seq_num=8000, min_len=50, max_len=100, max_hours=2, save_path=None, name_suffix=None,
                           min_df_length=150, min_df_alarm_type=4, previous_dataset=None
                           ):

    if ignore_group_id:
        df[group_col] = 0
    datasets = {}
    for i, alarm_df in df.groupby(group_col):
        # if len(alarm_df) < min_df_length or len(alarm_df[[alarm_type_col]].drop_duplicates(alarm_type_col)) <= min_df_alarm_type:
        #     continue

        index_map_df = alarm_df[[alarm_type_col]].drop_duplicates(alarm_type_col).reset_index(drop=True)
        index_map_df['ind'] = np.arange(len(index_map_df))
        if previous_dataset is not None:
            try:
                old_index_map = pd.read_csv(os.path.join(previous_dataset, i, 'index_map.csv'), index_col=None)
                index_map_df = update_index_df(old_index_map[['index', 'ind']], index_map_df, key_col='index', index_col='ind')
                print('Merging old index map and new index map successfully. From %d to %d' % (len(old_index_map), len(index_map_df)))
            except FileNotFoundError as e:
                print('%s is not found the previous dataset: %s' %
                      (os.path.join(previous_dataset, i, 'index_map.csv'), previous_dataset)
                      )

        # 使用 index_map_df 将 alarm_df 中告警类型列的值进行替换
        alarm_df = alarm_df.join(index_map_df.set_index(alarm_type_col), on=alarm_type_col, how='left')
        alarm_df[alarm_type_col] = alarm_df['ind']
        del alarm_df['ind']

        dataset, output_dir = generate_event_alarm_datasets(
            alarm_df, min(seq_num, len(alarm_df)), n_types=len(index_map_df), min_len=min_len, max_len=max_len,
            max_hours=max_hours, save_path=save_path, name='all' if ignore_group_id else i,
            alarm_type_col=alarm_type_col, name_suffix=name_suffix,
        )
        if dataset is None:
            continue

        datasets[i] = dataset, output_dir
        index_map_df.to_csv(
            os.path.join(output_dir, 'index_map.csv')
        )
        alarm_df.to_csv(os.path.join(output_dir, 'alarm.csv'))
    if ignore_group_id:
        return datasets[0]
    else:
        return datasets




