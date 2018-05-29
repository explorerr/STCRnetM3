#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 09:36:17 2018

@author: zhangrui
"""

import os
import math
import random
import _pickle as pickle
import numpy as np
import pandas as pd
from sklearn import preprocessing
import datetime


def add_month(start_year, start_month, add):

    start_month += add
    year = start_year + start_month / 12
    month = start_month % 12
    if month == 0:
        month = 12
        year -= 1
    return int(year), int(month)


def processing_raw_file(file_path):

    df = pd.read_csv(file_path, na_values=['NA', 'NULL', 'NaN', '\\N'])
    df.utc = [datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S") for x in df.utc]
    df['year'] = df.utc.apply(lambda x: x.year)  # [x.year for x in df.utc]
    df['month'] = df.utc.apply(lambda x: x.month)  # [x.month for x in df.utc]
    df['weekday'] = df.utc.apply(lambda x: x.weekday())  # [x.weekday() for x in df.utc]
    df['weeknum'] = df.utc.apply(lambda x: x.isocalendar()[1])  # [x.isocalendar()[1] for x in df.utc]
    df['iday'] = 0
    df['mark'] = 0
    df['gfs_lag1'] = np.NaN
    df['gfs_lag2'] = np.NaN
    df['gfs_lag3'] = np.NaN
    df = df.sort_values(by=['id', 'parameter', 'utc'])
    for param in set(df.parameter):
        for id in set(df.id):
            param_mask = (df.parameter == param) & (df.id == id)
            start_day = min(df.utc[param_mask])
            df.iday[param_mask] = [(x - start_day).days
                                   for x in df.utc[param_mask]]
            maxDay = max(df.iday[param_mask])
            # setting the last 90 days as testing days
            df.mark[(param_mask) & (df.iday > (maxDay - 90))] = 2
            # setting the same 90 days in previous year as validationg dataset
            df.mark[(param_mask) & (df.iday > (maxDay - 365 - 90)) & (df.iday < (maxDay - 365))] = 1
            df.gfs_lag1[param_mask] = df.gfs[param_mask].shift()
            df.gfs_lag2[param_mask] = df.gfs[param_mask].shift(2)
            df.gfs_lag3[param_mask] = df.gfs[param_mask].shift(3)

    df = df.drop_duplicates(['id', 'parameter', 'utc'])
    # randomly sample 20% of the last year data as validation set
    df = df.dropna(axis=0, how='any')

    return df


file_path = '~/git/data/CRD_combined_crop.csv'

df = processing_raw_file(file_path)


category_feature_list = ['id', 'parameter', 'year', 'month', 'weekday', 'weeknum']
label_enc_list = []

for category_feature in category_feature_list:
    label_enc = preprocessing.LabelEncoder()
    label_enc.fit(df.loc[:, category_feature])
    label_enc_list.append(label_enc)
    df.loc[:, category_feature] = label_enc.transform(df.loc[:, category_feature])


"""
Normalization of numeric data
"""
numeric_feature_list = ['gfs', "gfs_lag1", "gfs_lag2", "gfs_lag3"]

df[numeric_feature_list] = preprocessing.scale(df[numeric_feature_list])
with open('wea_bias_corr.pkl', 'wb') as f:
    pickle.dump([df, label_enc_list], f, -1)
