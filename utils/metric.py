import os
import sys
sys.path.append('..')

import pandas as pd
import numpy as np
from functools import partial
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, confusion_matrix, roc_curve
import random
from utils import PROJECT_ROOT_PATH
from utils.root_find import RootCauseFind


def df_list_handler(func, df_list):
    vs = []
    sum_length = 0
    for a, b in df_list:
        value = func(a, b)
        vs.append(value * len(a))
        sum_length += len(a)
    vs = np.stack(vs, axis=0).sum(axis=0)/sum_length
    return vs


def is_null(x):
    if isinstance(x, str) and len(x) > 0:
        return False
    if x is None or np.isnan(x):
        return True
    return False


def effective_intersection(df_a, col_a, df_b, col_b, only_derivative=False):

    not_null_index_a = df_a[~(df_a[col_a].isnull())].index
    not_null_index_b = df_b[~(df_b[col_b].isnull())].index
    effective_indexes = not_null_index_a.intersection(not_null_index_b)
    if only_derivative:
        derivative_index = df_a.loc[df_a.index != df_a['label-root-1']].index
        effective_indexes = effective_indexes.intersection(derivative_index)

    return df_a.loc[effective_indexes], df_b.loc[effective_indexes]



def acc_k(gt_df, detected_df, k=5, root_1_col='label-root-1', only_derivative=False):
    """

    :param gt_df: 人为标注的 DataFrame, 需要包含 label-root-1, label-root-2 ... label-root-x，x为任意值
    :param detected_df: 算法检测的DataFrame，需要包含 root-1, root-2....root-k, 以及root_alarm
    :return: direct_cause_ac, root_cause_ac, root_ac
    """
    assert type(gt_df) == type(detected_df)
    if isinstance(gt_df, list):
        return df_list_handler(func=partial(acc_k, k=k), df_list=zip(gt_df, detected_df))

    # assert len(gt_df) == len(detected_df)

    assert ('cause-%d' % k) in detected_df.columns

    gt_cols = ['label-root-%d' % i for i in range(1, 10) if ('label-root-%d' % i) in gt_df.columns]
    detected_cols = ['cause-%d' % i for i in range(1, k+1)]

    # root_gt_cols = ['label-root-%d' % i for i in range(1, 10) if ('label-root-%d' % i) in gt_df.columns]
    # root_detected_cols = ['cause-%d' % i for i in range(1, k+1)]

    result = np.zeros((k,))

    # gt_df, detected_df = effective_intersection(gt_df, root_1_col, detected_df, detected_cols[0], only_derivative=only_derivative)


    # Merely the accuracies of derivative events are considered. Root events are excluded.
    if only_derivative:
        gt_df = gt_df.loc[~gt_df['label_root_alarm']]
        detected_df = detected_df.loc[~detected_df['label_root_alarm']]

    for i, ((_id, gt), (d_id, detected)) in enumerate(zip(gt_df.iterrows(), detected_df.iterrows())):

        k_res = []
        for j in range(1, k+1):

            # region 统计直接因告警的正确率
            Vrc = set(map(int, filter(lambda x: x is not None and ~np.isnan(x), gt[gt_cols])))
            Rak = set(map(int, filter(lambda x: x is not None and ~np.isnan(x), detected[detected_cols[:j]])))

            if detected['root_alarm']:
                Rak.add(-1)

            direct_cause_ac = len(Vrc & Rak) / min(len(Vrc), len(Rak))
            k_res.append(direct_cause_ac)
        # endregion

        # region 统计根告警的正确率
        # Root_Vrc = set(gt_root.find_root(list(Vrc)).tolist())
        # Root_Rak = set(detected_root.find_root(list(Rak)).tolist())

        # Root_Vrc = set(filter(lambda x: isinstance(x, str) and len(x) > 0, gt[gt_cols]))
        # Root_Rak = set(filter(lambda x: isinstance(x, str) and len(x) > 0, detected[detected_cols]))

        # root_cause_ac = len(Root_Vrc & Root_Rak) / min(len(Root_Vrc), len(Root_Rak))
        # endregion

        # result += np.array([direct_cause_ac, root_cause_ac]) / len(gt_df)
        result += np.array(k_res) / len(gt_df)

    return result


def avg_k(gt_df, detected_df, k=5):
    """
    :param gt_df:
    :param detected_df:
    :param k:
    :return: direct_cause_avg, root_cause_avg
    """

    if isinstance(gt_df, list):
        return df_list_handler(func=partial(avg_k, k=k), df_list=zip(gt_df, detected_df))

    result = np.array([0, 0], dtype=float)
    for i in range(1, k+1):
        result += acc_k(gt_df, detected_df, i)/k
    return result


def root_identification_acc(gt_df, detected_df, root_1_col='label-root-1',
                            is_root_col='root_alarm', root_prob_col='modified_root_prob'):
    """
    评估根告警的检测精度和召回率
    """

    if isinstance(gt_df, list):
        return df_list_handler(
            func=partial(root_identification_acc, root_1_col=root_1_col),
            df_list=zip(gt_df, detected_df)
        )

    gt_df = gt_df.set_index('_id')
    detected_df = detected_df.set_index('_id')
    detected_df.loc[detected_df[root_prob_col].isnull(), root_prob_col] = 0.2

    # gt_df.loc[gt_df[root_1_col].isnull(), root_1_col] = gt_df.loc[gt_df[root_1_col].isnull()].index
    gt_df, detected_df = effective_intersection(gt_df, root_1_col, detected_df, 'root-1')
    is_root_alarm_gt = (gt_df.index == gt_df[root_1_col]).tolist()
    is_root_alarm_detected = (detected_df[is_root_col]).tolist()
    accuracy = accuracy_score(is_root_alarm_gt, is_root_alarm_detected)
    recall = recall_score(is_root_alarm_gt, is_root_alarm_detected)

    if root_prob_col not in detected_df.columns:
        detected_df.loc[detected_df[is_root_col], root_prob_col] = 1.0
        detected_df.loc[~detected_df[is_root_col], root_prob_col] = 0.0

    detected_score = (detected_df[root_prob_col]).tolist()

    roc_auc = roc_auc_score(is_root_alarm_gt, detected_score)

    fpr, tpr, _ = roc_curve(is_root_alarm_gt, detected_score)

    return np.array([accuracy, recall]), roc_auc, (fpr, tpr)



if __name__ == '__main__':

    def random_gt(df):
        """
        随机生成一个ground truth df，用于测试
        :param df: 利用 RootCauseDetectionBase.detect检测过的DataFrame
        :return: df内添加列: label-root-1, label-root-2 ... label-root-3
        """
        gt_df = df.copy()
        gt_df[['label-root-%d' % i for i in range(1, 4)]] = None
        for i in range(len(gt_df)):
            _id = gt_df.iloc[i].name
            if i == 0 or random.random() < 0.1:
                #第i行告警为根告警
                if random.random() < 0.9:
                    # 不标注默认为该告警是根告警
                    gt_df.loc[_id, 'label-root-1'] = _id
                continue

            for j in range(1, 4):
                gt_df.loc[_id, 'label-root-%d' % j] = gt_df.iloc[random.randint(max(0, i-5), i-1)].name
                if random.random() < 0.3:
                    break
        return gt_df[['label-root-%d' % i for i in range(1, 4)]]


    DATA_DIR = os.path.join(PROJECT_ROOT_PATH, 'outputs', 'merge_alarm')
    detected_df = [pd.read_excel(os.path.join(DATA_DIR, 'eOMC', 'compressed_eOMC.xlsx')).set_index('_id')]
    gt_df = [random_gt(df) for df in detected_df]
    root_alarm_identification_res = root_identification_acc(gt_df, detected_df)
    print('Acc: %.2f, Recall: %.2f\n' % (root_alarm_identification_res[0], root_alarm_identification_res[1]))
    acc = {}
    for i in range(1, 6):
        acc[i] = acc_k(gt_df, detected_df, k=i)
        print('AC@%d of direct: %.2f, AC@%d of root: %.2f' % (i, acc[i][0], i, acc[i][1]))
        v = np.array([v for _, v in acc.items()]).mean(axis=0)
        print('AVG@%d of direct: %.2f, AVG@%d of root: %.2f\n' % (i, v[0], i, v[1]) )
    # avg = avg_k(gt_df, detected_df, k=5)




