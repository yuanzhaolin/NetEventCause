import os.path

import pandas as pd
import numpy as np
import random
from collections import defaultdict

# class SequenceDataset:
#     def __init__(self, alarm_path):


class ObservationDataset:
    def __init__(self, alarm_type_index_df, df, group_identify=None, time_slice_col='belong'):
        self.group_identify = group_identify
        self.item_dict = defaultdict(lambda: -1)
        self.intermediate_set = []
        self.abbreviation = {}
        self.name2abbr = {}
        self.mat_01 = None
        self.alarm_snapshoot = []
        self.node_names = []
        self.item2external_index={}
        self.df = df

        ac_items = df[['alarm_type', 'ci_type']].drop_duplicates().values.tolist()
        for i, item in enumerate(alarm_type_index_df[['alarm_type', 'ci_type']].values.tolist()):
            if item in ac_items:
                _ = self.item2id('-'.join(item))
                self.item2external_index['-'.join(item)] = i

        for i, alarm_snapshot_df in df.groupby(time_slice_col):
            self.add_snapshot(alarm_snapshot_df)
        self.generate_01_mat()

    def add_snapshot(self, snapshot: pd.DataFrame):

        self.alarm_snapshoot.append(snapshot)
        existed_item = snapshot[['alarm_type', 'ci_type']].drop_duplicates().values.tolist()

        if len(existed_item) == 1 and existed_item[0][0] == '无告警':
            items = []
        else:
            items = ['##'.join(x) for x in existed_item]

        duration = (snapshot['close_timestamp']-snapshot['open_timestamp']).to_list()[0]/1e9

        self.intermediate_set.append(([self.item2id(x, auto_add=False) for x in items], duration))

    @property
    def nodes_num(self):
        return len(self.node_names)

    def item2id(self, item, auto_add=True):
        """
        名字 to id
        :param item: name ，如: 路由器告警告警##Router
        :param auto_add: 不存在时是否自动添加
        :return: int,
        """
        if item not in self.item_dict.keys() and auto_add:
            new_id = len(self.node_names)
            self.item_dict[item] = new_id
            self.abbreviation[f'X{str(new_id)}'] = item
            self.name2abbr[item] = f'X{str(new_id)}'
            self.node_names.append(item)

        return self.item_dict[item]

    def generate_01_mat(self):
        if len(self.item_dict.keys()) == 0:
            self.mat_01 = np.array([[0]], dtype=np.int64), np.array([1.0])
            return

        p_list = []
        discrete_set = []
        for row, p in self.intermediate_set:
            res = [0 for _ in range(len(self.item_dict))]
            for i in row:
                res[i] = 1
            discrete_set.append(res)
            p_list.append(p)

        p_list = np.array(p_list) / sum(p_list)
        self.mat_01 = np.array(discrete_set, dtype=np.int64),  p_list

    def sample_01(self, num=None):
        mat, p_list = self.mat_01
        return mat[np.random.choice(len(mat), len(self.item_dict) if num is None else num, p=p_list)]

    @property
    def names(self):
        return self.node_names

    def sample_snapshot(self, num=1, alarm_existed=True, min_number=1, belong_id=None):
        """
        采样num个告警集，每个告警集中的告警数量不小于min_number，如果belong_id不为空，直接范围指定belong_id的告警集。
        :param num:
        :param alarm_existed:
        :param min_number:
        :param belong_id:
        :return:
        """
        if belong_id is not None:
            return [self.df.loc[self.df['belong'] == belong_id]]

        def f(df):
            """
            判断df是否存在类型为 "无告警"的告警
            :param df:
            :return:
            """
            return ~(df['alarm_type'] == '无告警').all()

        def num_filter(df):
            return len(df) >= min_number

        mat, p_list = self.mat_01
        res = []

        def alarm_snap_iterator():

            while True:
                for i in np.random.choice(len(self.alarm_snapshoot), num-len(res), p=p_list):
                    tmp_res = self.alarm_snapshoot[i]
                    if num_filter(tmp_res) and (not alarm_existed or f(tmp_res)):
                        yield tmp_res

        res = [x for _, x in zip(range(num), alarm_snap_iterator())]
        return res


def group_df2observation_dataset(
        df: pd.DataFrame, alarm_type_index_df, ignore_group_id=False, time_slice_col='belong', group_col='alarm_group_id'
):
    """
    根据group_col的取值，生成多个ObservationDataset

    :param df:
    :param alarm_type_index_df:
    :param ignore_group_id:
    :param time_slice_col:
    :param group_col:
    :return:
    """

    if ignore_group_id:
        def delete_no_alarm_items(d):
            if (d['alarm_type'] == '无告警').all():
                d = d.iloc[0:1]
            else:
                d = d.loc[d['alarm_type'] != '无告警']
            return d

        df = pd.concat(
            [delete_no_alarm_items(d) for _, d in df.groupby('belong')]
        )

        df[group_col] = 0

    alarm_datasets = {
        # g_id: AlarmDataset(group_identify=g_id) for g_id in df[group_col].drop_duplicates().to_list()
        g_id: ObservationDataset(
            alarm_type_index_df=alarm_type_index_df,
            df=df.loc[df[group_col] == g_id],
            group_identify=g_id,
            time_slice_col=time_slice_col
        )
        for g_id in df[group_col].drop_duplicates().to_list()
    }

    if ignore_group_id:
        return alarm_datasets[0]
    else:
        return alarm_datasets


# if __name__ == '__main__':
#
#     tmp_data_path = os.path.join('..', 'dataset', 'grouped_alarm', '0.csv')
#
#     topo_group_id_df = pd.read_csv(
#         os.path.join('..', 'dataset', 'ci_cluster.csv')
#     )
#
#     alarm_group_id_df = pd.read_csv(
#         os.path.join('..', 'dataset', 'alarm_cluster.csv')
#     )
#
#     # tmp_data = os.path.join('..', 'dataset', 'alarm_tmp.csv')
#     # grouped_alarm_path = os.path.join('..', 'dataset', 'grouped_alarm')
#     df = pd.read_csv(tmp_data_path)
#     # discrete_set, item_dict = groupdf2discrete(
#     #     df=None, dir='../dataset/grouped_alarm_new', topo_group_id_df=topo_group_id_df,
#     #     alarm_group_id_df=alarm_group_id_df
#     # )
#     #
#
#     alarm_datasets = group_df2datasets(
#         df=pd.read_csv(tmp_data_path)
#     )
#
#     # alarm_datasets = groupdf2discrete(
#     #     df=None, dir='../dataset/grouped_alarm_new', topo_group_id_df=topo_group_id_df,
#     #     alarm_group_id_df=alarm_group_id_df
#     # )
#     for _, ad in alarm_datasets.items():
#         assert isinstance(ad, AlarmDataset)
#         print('group: %d' % _, ad.sample_01(10))
#     # os.makedirs(grouped_alarm_path, exist_ok=True)
