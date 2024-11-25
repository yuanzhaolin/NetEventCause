
import datetime
import os
import shutil

import pandas as pd
import random
import numpy as np
from collections import defaultdict
import time

PROJECT_ROOT_PATH = os.path.abspath(
    os.path.dirname(os.path.dirname(__file__))
)


def ip_substitude(df, col):
    df[col] = df[col].str.replace(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', '*')


def find_CI_by_id(df, id):
    return df.loc[df['id'] == id]


def latest_update_time(path):
    mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(path))
    return mod_time

def find_dir_until(path, file):
    if file in os.listdir(path):
        return path
    sub_dirs = [os.path.join(path, x) for x in os.listdir(path)]
    sub_dirs = list(filter(os.path.isdir, sub_dirs))
    sub_dirs = sorted(sub_dirs, key=latest_update_time, reverse=True)

    if len(sub_dirs) == 0:
        return None

    for x in sub_dirs:
        res = find_dir_until(x, file)
        if res is not None:
            return res

    return None


def move_column_to_front(df, columns):
    """
       将DataFrame的某些列提到最前面
       :param df: DataFrame对象
       :param columns: 需要提到最前面的列名列表
       :return: 提到最前面后的DataFrame对象
    """
    if not isinstance(columns, list):
        columns = [columns]

    # 将需要提到最前面的列名列表与原始列名列表合并
    new_columns = columns + [col for col in df.columns if col not in columns]
    # 使用reindex函数重新排列列顺序
    return df.reindex(columns=new_columns)


def reset_dir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path, exist_ok=True)

def filename_from_path(path):
    return os.path.split(path)[1].split('.')[0]

def merge_numeric_csv_in_dir(path, excluded_file=None, included_file=None, max_file_num=int(1e9)):
    if excluded_file is None:
        excluded_file = []
    if not isinstance(excluded_file, list):
        excluded_file = [excluded_file]

    df_list = []
    file_list = os.listdir(path)
    random.shuffle(file_list)

    if included_file is None:
        included_file = file_list

    for i, file in enumerate(file_list):
        if not file.split('.')[0].isnumeric() or file in excluded_file:
            continue
        if included_file is not None and file not in included_file:
            continue
        df_list.append(pd.read_csv(os.path.join(path, file)))
        if len(df_list) >= max_file_num:
            break
    merged_df = pd.concat(df_list)
    return merged_df
    # merged_df.to_csv(os.path.join(path, 'all.csv'))


class TimeRecorder:
    def __init__(self):
        self.infos = defaultdict(float)

    def __call__(self, info, *args, **kwargs):
        class Context:
            def __init__(self, recoder, info):
                self.recoder = recoder
                self.begin_time = None
                self.info = info

            def __enter__(self):
                self.begin_time = time.time()

            def __exit__(self, exc_type, exc_val, exc_tb):
                self.recoder.infos[self.info] += time.time() - self.begin_time

        return Context(self, info)

    def __str__(self):
        return ' '.join(['{}: {:.2f}s'.format(info, t) for info, t in self.infos.items()])

    def __getitem__(self, item):
        return self.infos[item]


# def adj_list2adj_mat(edges, n):

class SimpleLogger(object):
    def __init__(self, f, header='#logger output'):
        dir = os.path.dirname(f)
        self.dir = dir
        self.begin_time_sec = time.time()
        # print('temp dir', dir, 'from', f)
        if dir != '' and not os.path.exists(dir):
            os.makedirs(dir)
        with open(f, 'w') as fID:
            fID.write('%s\n' % header)
        self.f = f

    def __call__(self, *args):
        # standard output
        print(*args)
        # log to file
        try:
            with open(self.f, 'a') as fID:
                fID.write('Time_sec = {:.1f} '.format(time.time() - self.begin_time_sec))
                fID.write(' '.join(str(a) for a in args) + '\n')
        except:
            print('Warning: could not log to', self.f)


def add_label_cols(df):
    df = df.reset_index()
    df[['label-root-%d' % i for i in range(1, 5)]] = None
    cols = list(df.columns)
    cols = ['_id']+['label-root-%d' % i for i in range(1, 5)] + [col for col in cols if not col.startswith('label') and col!='_id']
    df = df.reindex(columns=cols)
    return df


def update_index_df(old_index_df, new_index_df, key_col, index_col):
    """
    用于增量数据集构建，找到new_index_df中存在但在old_index_df中不存在的key，添加到old_index_df中，并赋予新的index编号。

    :param old_index_df:
    :param new_index_df:
    :return:
    """
    if not isinstance(key_col, list):
        key_col = [key_col]

    old_index_df = old_index_df.set_index(key_col, drop=True)
    # new_index_df = new_index_df.set_index(key_col, drop=True)

    res_index_df = old_index_df.copy()
    res_index_df['new'] = False
    if 'count' in res_index_df.columns:
        res_index_df['count'] = 0

    del new_index_df[index_col]
    new_index_df = new_index_df.join(
        old_index_df[[index_col]], how='left', on=key_col, rsuffix='old_'
    )

    extra_index_df = new_index_df.loc[new_index_df[index_col].isnull()]
    intersect_index_df = new_index_df.loc[~(new_index_df[index_col].isnull())]

    if 'count' in res_index_df.columns:
        intersect_index_df = intersect_index_df.set_index(key_col, drop=True)
        res_index_df['count'] = intersect_index_df['count']

    cur_max_index = old_index_df[index_col].max()
    extra_index_df[index_col] = np.arange(len(extra_index_df)) + cur_max_index + 1
    extra_index_df['new'] = True
    res_index_df = pd.concat([res_index_df.reset_index(), extra_index_df])

    return res_index_df


def dict2obj(my_dict):
    class MyObject:
        pass

    my_object = MyObject()
    for key, value in my_dict.items():
        setattr(my_object, key, value)
    return my_object


def file2df(file_path: str):
    if file_path.endswith('.xlsx'):
        return pd.read_excel(file_path, engine='openpyxl')
    elif file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    else:
        raise NotImplementedError


def mergedir_files2df(dir_path: str, cols=None):
    def add_cols(df, cols):
        df = df.copy()
        if cols is None:
            return df
        for c in cols:
            if c not in df.columns:
                df[c] = None
        return df

    res = []
    for file in os.listdir(dir_path):
        if file.endswith('xlsx') or file.endswith('csv'):
            df = file2df(os.path.join(dir_path, file))
            df = add_cols(df, cols)
            res.append(df[cols])

    return pd.concat(res)

