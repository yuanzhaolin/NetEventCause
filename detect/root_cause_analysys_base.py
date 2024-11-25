import sys


import pandas as pd
import numpy as np
from itertools import chain
from tqdm import tqdm
from utils.root_find import RootCauseFind
from utils import move_column_to_front, TimeRecorder


class RootCauseDetectionBase:
    def __init__(self, top_k=5, algorithm='Base', max_window_length=200,
                 window_moving_distance=80, **kwargs):
        self.top_k = top_k
        self.algorithm = algorithm
        self.max_window_length = max_window_length
        self.window_moving_distance = window_moving_distance

    def get_root_cause_cols(self):
        # root_cause_cols = list(
        #     chain(*[['root-%d' % (i+1), 'root-%d-score' % (i+1)] for i in range(self.top_k)])
        # ) + ['root_alarm', 'root_prob']

        root_cause_cols = list(
            chain(*[['cause-%d' % (i+1), 'cause-%d-score' % (i+1)] for i in range(self.top_k)])
        ) + [
            'root-%d' % (i+1) for i in range(self.top_k)
        ] + [
            'root_alarm', 'root_prob', 'modified_root_prob', 'identified', 'depth'
        ]


        return root_cause_cols

    # def final_return_cols(self):
    #     cols = ['alarm_type', 'ci_type', 'alarm_description', 'incident_open_time', 'incident_close_time']
    #     cols += self.get_root_cause_cols()
    #     cols += ['parent', 'detect_type', 'algorithm']
    #     cols += ['alarm_title', 'ci_name', 'ci_id', 'alarm_code', 'alarm_type_group', 'index',
    #              'compress_group', 'repeated', 'identified']
    #     return cols

    def detect(self, df: pd.DataFrame):
        """
        以dataframe的形式给定当前告警集，在df中添加列多个列

        algorithm: 检测算法，由__init__函数给定;
        detect_type:
            cause: 利用因果算法检测根因
            inherit: 利用告警压缩执行根因继承
            unknown: 无法识别

        parent: 当detect_type=inherit时，即依赖于继承时候，被继承alarm的_id

        root-x: 排名为x的根告警id
        root-x-score: 排名为x的根告警id对应的得分


        :param df:
        :param group_col:
        :param do_compress: 是否在诊断之前做压缩
        :return:
        """
        self.check_input(df)

        # rc_cols = self.get_root_cause_cols()

        final_results = self.init_result_df(df)

        next_detected_position = 0
        # while next_detected_position < len(result):
        #     start = max(0, next_detected_position - )

        print('Identifying root causes of %d alarms.' % len(df))
        all_reserved_windows = []

        for start in tqdm(range(
                0,
                len(df),
                self.window_moving_distance
        )):
            window_length = min(len(df)-start, self.max_window_length)
            end = start + window_length
            if next_detected_position >= end:
                continue

            print('Starting detection of window %d-%d' % (start, end))
            tr = TimeRecorder()
            with tr('window'):
                result_window = self._detect(df.iloc[start: end])

            print('Detecting time of window %d-%d: %.2f s' % (start, end, tr['window']))

            updated_rows = end - next_detected_position
            result_window = result_window.iloc[-updated_rows:]

            all_reserved_windows.append(result_window)

            next_detected_position = end

        result_window = pd.concat(all_reserved_windows)
        root_table = RootCauseFind(result_window, root_col='cause-1')
        for k in range(1, 1+self.top_k):
            result_window['root-%d' % k] = root_table.find_root(result_window['cause-%d' % k].tolist()).tolist()

        print('Root alarms identification ends.')

        return result_window

    def _detect(self, df: pd.DataFrame):
        raise NotImplementedError

    def check_input(self, df):
        assert df['t'].is_monotonic_increasing

    def init_result_df(self, df: pd.DataFrame):

        result = df.copy()
        # df = self.alarm_compress(df)

        rc_cols = self.get_root_cause_cols()
        result[rc_cols] = None

        result['parent'] = None
        result['algorithm'] = self.algorithm
        result['root_prob'] = None
        result['modified_root_prob'] = None
        result['root_alarm'] = False
        result['identified'] = True
        result['depth'] = 0

        result = move_column_to_front(result, rc_cols)

        # result = move_column_to_front(result, ['alarm_type', 'ci_type', 'alarm_description', 'incident_open_time', 'incident_close_time'])
        result = move_column_to_front(result, ['type', 't', 'label-root-1'])

        return result

    @property
    def single_cause_col_num(self):
        """
        目前对于每个root cause，只标识id和score，所以列数是2

        :return:
        """
        return 2
