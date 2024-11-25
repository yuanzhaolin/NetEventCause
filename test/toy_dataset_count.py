#!/usr/bin/python
# -*- coding:utf8 -*-


import pandas as pd

df = pd.read_csv('../cache/toy/dataset/ggem-1K-5/seq_demo.csv')

for i in range(0, df['type'].max()+1):
    num = len(df.loc[df['type']==i])
    root_num = len(df.loc[(df['type']==i) & (df['cause']==-1)])

    print('type %d: %d, root: %d, ratio: %.2f' % (i, num, root_num, root_num/num))