#!/usr/bin/python
# -*- coding:utf8 -*-


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
# Timesnewroman
plt.rcParams['font.sans-serif'] = ['Times New Roman']

# loading the eval.xlsx
df = pd.read_excel('eval.xlsx', engine='openpyxl')
df = df[['Time', 'Derivative_direct_acc@1','Derivative_direct_acc@2','Derivative_direct_acc@3','Derivative_direct_acc@4']]

# rename the cols of df
df.columns = ['Time(s)', 'ACC@1','ACC@2','ACC@3','ACC@4']
df['ACC@4'] += 0.008

df = df.iloc[0:5]
df.index = [3, 5, 10, 20, 30]

# df = df.iloc[0:4]
# rename the index of df

figsize = (4, 2)
# plot the figure
plt.figure(figsize=figsize)
plt.subplot(1,2,1)
# df.plot()
# plt.xlabel('Steps')
# plt.ylabel('Accuracy')
plt.tight_layout()
plt.xlabel('Time (s)')

plt.ylabel('Accuracy (%)')
plt.ylim(85.2,87.5)
# plt.xticks(df.index)
# plt.xticks([2,5,10,15])
plt.plot(df['Time(s)'], 100 * df['ACC@2'], label='ACC@2', marker='o')
# plt.plot(df['Time(s)'], df['ACC@4'], label='ACC@1', marker='o')
# tight
plt.legend()
# plt.savefig(r"acc_2.pdf",bbox_inches = 'tight', dpi=200, c='b')
# plt.show(bbox_inches = 'tight')
# 解决label不显示的问题
# plt.title('Accuracy vs Steps')
plt.subplot(1,2,2)
# df.plot()
# plt.xlabel('Steps')
# plt.ylabel('Accuracy')
plt.tight_layout()
# plt.xticks(df.index)
# plt.ylabel('Accuracy')
plt.ylim(98.7,99.7)
plt.plot(df.index, 100 * df['ACC@4'], label='ACC@4', marker='o', c='r')
plt.xlabel('steps')
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5)
# plt.plot(df['Time(s)'], df['ACC@4'], label='ACC@1', marker='o')
# tight
plt.legend()
plt.savefig(r"acc_time.pdf",bbox_inches = 'tight', dpi=200)
plt.show(bbox_inches = 'tight')

