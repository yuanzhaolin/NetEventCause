
import json
import os.path
import pandas as pd
import yaml
import torch
from utils import PROJECT_ROOT_PATH, TimeRecorder
from collections import defaultdict
# from cause.event.pkg.models

from cause.event.pkg.models.ode_rnn import ODERecurrentPointProcess
# from cause.event.pkg.models.rnn import ExplainableRecurrentPointProcess
from cause.event.pkg.models.rnn import ExplainableRecurrentPointProcess
# from cause.compress.rules import CompressRule
from .root_cause_analysys_base import RootCauseDetectionBase
import numpy as np
from cause.dataset.temporal_events_dataset import df2events_seq
from cause.compress import str_similar



class NearestEventCauseDetection(RootCauseDetectionBase):
    def __init__(self,
                 prior_graph=None,
                 **kwargs
                 ):

        super(NearestEventCauseDetection, self).__init__(algorithm='nearest', **kwargs)

        if prior_graph is None:
            prior_graph = {
                0: (-1, 0),
                1: (0, 10),
                2: (0, 20),
                3: (2, 20),
                4: (-1, 0)
            }
        self.prior_graph = prior_graph

    def _detect(self, event_df: pd.DataFrame,  group_col='compress_group'):

        # if not compressed:
        #     raise NotImplementedError('df should be ')

        # df = df.set_index('_id')

        result = self.init_result_df(event_df)

        rc_cols = self.get_root_cause_cols()
        # result['detect_type'] = None

        not_identified_rows = event_df['type'].isnull()
        result.loc[not_identified_rows, rc_cols[0]] = result.loc[not_identified_rows].index
        result.loc[not_identified_rows, rc_cols[1]] = np.inf
        result.loc[not_identified_rows, 'identified'] = False
        result.loc[not_identified_rows, 'root_alarm'] = True
        result.loc[not_identified_rows, 'root_prob'] = 1.0
        result.loc[not_identified_rows, 'modified_root_prob'] = 1.0

        # endregion

        event_df = event_df.loc[~not_identified_rows]

        event_df['depth'] = 0


        # log_intensities_events.detach().cpu(), prior_log_intensities_events.detach().cpu()

        tr = TimeRecorder()
        # with tr('attribution'):
        #     contribution_mat, log_intensities_events, base_log_intensities_events, prior_log_intensities_events = [x.squeeze(dim=0) for x in self.model.get_seq_contribution(
        #         data_batch.to(self.device)
        #     )]

        time_record = defaultdict(list)


        print('Scanning time: %.2f s' % (tr['attribution']))
        for i in range(len(event_df)):

            cols = []
            res = []
            cur_event = event_df.iloc[i]

            root_cause_rank = []

            score = float(self.top_k)

            for cause_event in reversed(time_record[self.prior_graph[cur_event['type']][0]]):
                if cur_event['t'] - cause_event['t'] > self.prior_graph[cur_event['type']][1]:
                    break

                root_cause_rank.append((score, cause_event['pos']))
                score -= 1.0
                # if str_similar(cur_event['alarm_description'], cause_event['alarm_description']) > 0.5:
                #     time_record[self.prior_graph[cur_event['type']][0]].remove(cause_event)
                #     break

            if len(root_cause_rank) == 0:
                root_alarm_prob = 1.0
                modified_root_prob = 1.0
            else:
                root_alarm_prob = 0
                modified_root_prob = 0.0

            cols += ['root_prob']
            res += [root_alarm_prob]

            root_cause_rank = filter(lambda x: x[0] > 0, root_cause_rank)
            root_cause_rank = sorted(root_cause_rank, reverse=True)
            causative_alarms = [event_df.iloc[i] for _, i in root_cause_rank[:self.top_k]]

            cols += ['modified_root_prob']
            res += [modified_root_prob]

            cols += ['root_alarm', 'depth']

            if modified_root_prob > 0.5:
                # 识别 i 为根告警
                # result.loc[compressed_df.iloc[i].name, 'root_alarm'] = True
                # root_cause_rank = [(np.inf, i)] + root_cause_rank
                event_df.loc[event_df.iloc[i].name, 'depth'] = 0
                res += [True, 0]
            else:
                event_df.loc[event_df.iloc[i].name, 'depth'] = causative_alarms[0]['depth'] + 1
                res += [False, causative_alarms[0]['depth'] + 1]

            for j, cause in zip(range(self.top_k), root_cause_rank):

                cause_id_col = rc_cols[j * self.single_cause_col_num]
                cause_score_col = rc_cols[j * self.single_cause_col_num + 1]
                cols += [cause_id_col, cause_score_col]
                res += [event_df.iloc[cause[1]].name, float(cause[0])]

            result.loc[event_df.iloc[i].name, cols] = res

            time_record[cur_event['type']].append({'t': cur_event['t'], 'pos': i})

        return result

    def cause_score_modify(self, score, cause_alarm, current_alarm, T=3600, ratio=5.0):
        """
        return : [0,inf] 如果返回值为0，将不作为因告警集中的候选告警
        """
        score = max(score, 1e-2)
        if current_alarm['type']==1 and cause_alarm['type']!=0:
            score = 0
        elif current_alarm['type']==2 and cause_alarm['type']!=0:
            score = 0
        elif current_alarm['type']==3 and cause_alarm['type']!=2:
            score = 0
        elif current_alarm['type']==4 or current_alarm['type']==0:
            score = 0

        return score
    def events_df2tensor_batch(self, df):
        event_seq = df2events_seq(df, alarm_type_col='type', time_col='t')
        data_batch = torch.FloatTensor(np.array(event_seq)).unsqueeze(dim=0)
        return data_batch
