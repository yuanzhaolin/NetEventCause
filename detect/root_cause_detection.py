import sys

import pandas as pd
import os
from itertools import chain
import numpy as np
from .alarm_event_graph import AlarmEventGraphFile, AlarmEventGraphNeo4j, AlarmEventGraphBase, AlarmEventGraphScoreMat
from .random_walk import RandomWork


class RootCauseDetection:
    def __init__(self, alarm_graph_event: AlarmEventGraphBase,  top_k=5, algorithm='random walk',
                 coefficient_file_path=None):
        self.top_k = top_k
        self.algorithm = algorithm
        self.age = alarm_graph_event
        if coefficient_file_path is None:
            coefficient_file_path = os.path.join(os.path.dirname(__file__), 'data', 'coefficient.csv')
        self.coeff_table = pd.read_csv(coefficient_file_path).set_index(['a', 'b'])
        self.random_walk = RandomWork()

    def get_root_cause_cols(self):
        root_cause_cols = list(
            chain(*[['root-%d' % (i+1), 'root-%d-score' % (i+1)] for i in range(self.top_k)])
        )
        return root_cause_cols

    def detect(self, df: pd.DataFrame):
        """
        以dataframe的形式给定当前告警集，在df中添加列多个列

        :param df:
        :return:
        """

        result = df.copy().set_index('_id')

        root_cause_cols = self.get_root_cause_cols()
        result[root_cause_cols] = None
        result['parent'] = None

        core_alarms = []
        for key, df in result.sort_values('incident_open_timestamp').groupby(['alarm_type', 'ci_type']):
            first_alarm = df.iloc[0]
            if self.age.alarm_ci_name(first_alarm['alarm_type'], first_alarm['ci_type']) in self.age.nodes:
                core_alarms.append(first_alarm)
                result.loc[first_alarm.name, 'algorithm'] = self.algorithm
                result.loc[first_alarm.name, 'detect_type'] = 'cause'
            else:
                result.loc[first_alarm.name, 'algorithm'] = self.algorithm
                result.loc[first_alarm.name, 'detect_type'] = 'unknown'

            for i in range(1, len(df)):

                alarm = df.iloc[i]
                result.loc[alarm.name, 'algorithm'] = self.algorithm
                result.loc[alarm.name, 'detect_type'] = 'inherit'
                result.loc[alarm.name, 'parent'] = first_alarm.name

        core_alarms_df = pd.DataFrame(core_alarms)
        core_alarms_df = self.root_cause_analysis(core_alarms_df)

        # 按照index进行赋值
        result[root_cause_cols] = core_alarms_df[root_cause_cols]

        result.loc[result['detect_type'] == 'inherit', root_cause_cols] = result.loc[
            result.loc[result['detect_type'] == 'inherit', 'parent'], root_cause_cols
        ].values
        return result

    def root_cause_analysis(self, df):

        root_cause_cols = self.get_root_cause_cols()
        single_cause_col_num = len(root_cause_cols) // self.top_k

        # 默认赋值为告警自己的_id
        df[root_cause_cols[0]] = df.index
        df[root_cause_cols[1]] = 1.0

        num_nodes = len(df)
        nodes, adj_list = self.age.generate_alarm_event_graph(df)
        mat = np.zeros((num_nodes, num_nodes))
        for i, s in enumerate(nodes):
            for j, e in adj_list[i]:
                if e['conn']:
                    mat[i, j] = 1

        coeffs = []
        for i, s in enumerate(nodes):
            coeffs.append(
                [self.coeff_table.loc[(s['node_name'], e['node_name'])]['statistic'] for e in nodes]
            )
        coeffs = np.array(coeffs)
        coeffs = np.clip(coeffs, 0, None)

        coeffs[np.diag_indices_from(coeffs)] = 0.2

        for i, s in enumerate(nodes):
            steady, acc_visits = self.random_walk.run(mat, coeff=coeffs[i], inverse=True, ab_node=i)
            root_cause_rank = [(score, k) for k, score in enumerate(acc_visits)]
            root_cause_rank = sorted(root_cause_rank, reverse=True)
            s_open_time = s['incident_open_timestamp']

            for j, cause in zip(range(self.top_k), root_cause_rank):

                if cause[0] < 1e-6 and cause[1] != i:
                    # score 已经很低了，而且不是本身，说明第j个以及之后的告警都不是i的潜在根告警
                    break
                if df.iloc[cause[1]]['incident_open_timestamp'] > s_open_time:
                    continue
                cause_id_col = root_cause_cols[j * single_cause_col_num]
                cause_score_col = root_cause_cols[j * single_cause_col_num + 1]

                df.loc[s['_id'], cause_id_col] = df.iloc[cause[1]].name
                df.loc[s['_id'], cause_score_col] = cause[0]
            pass

        return df


