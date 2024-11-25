import sys
import os, bson
import copy
import numpy as np
import pandas as pd
from utils import ip_substitude


reserved_cols = [
    "_id",  "alarm_title", "alarm_type",
    "alarm_description", "alarm_level", 'status', 'alarm_type_group',
    "ci_name", "ci_type", "ci_id", "incident_open_time", "incident_close_time", "alarm_code"
]

def merge_and_generate_type(alarm_df: pd.DataFrame, ci_df: pd.DataFrame, alarm_type_norm: pd.DataFrame=None):
    # tmp_df = alarm_df.join(type_association_df[['ci_type', 'ci_type_ci']].set_index('ci_type'), how='left',
    #                        on='ci_type').join(
    #     ci_df.rename(columns={'type': 'ci_type_ci', 'name': 'ci_name'}).set_index(['ci_name', 'ci_type_ci']),
    #     on=['ci_name', 'ci_type_ci'], how='inner', rsuffix='_ci')

    alarm_df = alarm_df.join(ci_df.set_index(['name', 'subtype'])['id'], how='left', on=['ci_name', 'ci_type'], rsuffix='_ci')
    if 'id_ci' in alarm_df.columns:
        alarm_df = alarm_df.rename(columns={'id_ci': 'ci_id'})
    else:
        alarm_df = alarm_df.rename(columns={'id': 'ci_id'})

    if alarm_type_norm is not None:
        alarm_type_association_df = alarm_df.join(
            alarm_type_norm[['alarm_title', 'alarm_type']].set_index('alarm_title'), on='alarm_title', how='left'
        )

        alarm_type_association_df.loc[
            alarm_type_association_df['alarm_type'].isnull(), 'alarm_type'
        ] = alarm_type_association_df.loc[alarm_type_association_df['alarm_type'].isnull(), 'alarm_title']
    else:

        alarm_df['alarm_type'] = alarm_df['alarm_title']
        alarm_type_association_df = alarm_df

    ip_substitude(alarm_type_association_df, 'alarm_type')

    if 'alarm_type_group' not in alarm_type_association_df.columns:
        alarm_type_association_df['alarm_type_group'] = None
    alarm_type_association_df = alarm_type_association_df[reserved_cols]
    alarm_type_association_df = preprocess_alarm(alarm_type_association_df)
    return alarm_type_association_df


def preprocess_alarm(alarm_df):
    """
    扩种alarm字段: 时间戳, 持续时间
    :param alarm_df:
    :return:
    """
    alarm_df['incident_open_timestamp'] = pd.to_datetime(alarm_df['incident_open_time']).values.astype(np.int64)

    # todo 临时处理，要求每个alarm的持续时间不超过2天，避免快照过大。
    alarm_df['incident_close_timestamp'] = pd.to_datetime(alarm_df['incident_close_time']).values.astype(np.int64)
    alarm_df['incident_close_timestamp'] = np.min(
        [pd.to_datetime(alarm_df['incident_close_time']).values.astype(np.int64), alarm_df['incident_open_timestamp'] + int(1e9) * 3600 * 24],
        axis=0
    )
    alarm_df['incident_duration_timestamp'] = alarm_df['incident_close_timestamp'] - alarm_df['incident_open_timestamp']
    alarm_df['incident_duration_time'] = pd.to_datetime(alarm_df['incident_close_time']) - pd.to_datetime(alarm_df['incident_open_time'])
    return alarm_df
