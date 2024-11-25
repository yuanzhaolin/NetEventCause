
import pandas as pd
import numpy as np
import os
import sys
sys.path.append('..')
from utils import merge_numeric_csv_in_dir


def filter_time(df, percent=0.98):
    grouped = df.groupby('alarm_type')
    p = grouped['incident_duration_timestamp'].quantile(percent)

    def f(x):
        res = x['incident_open_timestamp'] < x['incident_close_timestamp'] and x['incident_duration_timestamp'] < p[x['alarm_type']]
        return res
    filtered_df = df[df.apply(f, axis=1)]
    return filtered_df


def filter_counts(df, counts_threshold=50):
    grouped = df.groupby('alarm_type')
    new_df = grouped.filter(lambda x: x['_id'].count() > counts_threshold)
    return new_df


def filter_level(df, lowest_level=3):
    df = df.loc[df['alarm_level'] <= lowest_level]
    return df


def filter_alarm(df, percent=0.98, counts_threshold=15, level_threshold=3):
    df = filter_counts(df, counts_threshold)
    # df = filter_time(df, percent)
    df = filter_level(df, level_threshold)
    return df


def extract_alarm_type_index(alarm_df):
    alarm_df_types = alarm_df[['alarm_type', 'ci_type']].drop_duplicates()
    alarm_df_types = alarm_df_types.reset_index()[['alarm_type', 'ci_type']].reset_index()
    # # alarm_df_types.to_csv(os.path.join(DATA_ROOT, 'alarm_type_index_old.csv'))
    # alarm_df_types = pd.read_csv(os.path.join(DATA_ROOT, 'alarm_type_index_old.csv'))
    # alarm_df_types
    return alarm_df_types
