#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 09:36:17 2018

@author: zhangrui
"""

import os, math, random
import _pickle as pickle 
import numpy as np
import pandas as pd
from sklearn import preprocessing

def add_month(start_year, start_month, add):
    
    start_month += add
    year = start_year + start_month/12
    month = start_month%12
    if month == 0:
         month = 12
         year -= 1
    return int(year), int(month)

def processing_raw_file(file_path):
    ## read in raw file
    df = pd.read_csv(file_path, na_values=['NA','NULL','NaN','\\N'])
    df = df.rename( index=str, columns={"Series":"series", "N":"n","NF":"nf",
                 "Category":"cate","Starting Year":"year",
                 "Starting Month":"month"})
    df.year = df['year'].replace(0,1990)
    df.month = df['month'].replace(0,1)
    
    ## reshape the table as verticle table
    newdf = pd.melt(df, id_vars=['series',"n","nf","cate","year","month"])
    newdf = newdf.dropna(axis=0, how='any')
    newdf.variable = pd.to_numeric(newdf.variable)
    newdf = newdf.sort_values(by=['series','variable'])
    newdf = newdf.rename( index= str, columns={'variable':'iMonth','value':'qtty'})
    newdf.month = newdf.iMonth % 12
    newdf.month = newdf['month'].replace(0, 12)
    newdf.year = newdf.year + (newdf.iMonth-1) // 12
    
    ## mark training and testing data as 0 and 2 (will set valiation later as 1)
    ## nf: number of forecast, all set as 18 (last one year and a half)
    newdf['mark'] = 0
    newdf.mark.loc[newdf.iMonth + newdf.nf - newdf.n >0] = 2
########################
# For every year and the testing set, take 12 month as a period
# (1) For every test dataset, take the lag 12 month as a period, generated the lagMedian12, lagMedian6, lagMedian3, lagMedian1,
    #lagMean12, lagMean6, lagMean3
########################
    newdf['period'] = 0
    for s in pd.unique(newdf.series):
        mask = (newdf.series == s)
        max_period = newdf.loc[mask,'period'].shape[0] //12
        newdf.loc[mask,'period'] = max_period - newdf.iMonth[mask] // 12 + 1
    
    newdf0 = newdf
    newdf = newdf.join(pd.DataFrame(
    {
        'lagMedian12': np.nan,
        'lagMean12': np.nan,
        'lagMedianFirst3': np.nan,
        'lagMeanFirst3': np.nan,
        'lagMedian6': np.nan,
        'lagMean6': np.nan,
        'lagMedian4': np.nan,
        'lagMean4': np.nan,
        'lagMedian3': np.nan,
        'lagMean3': np.nan,
        'lagMean2': np.nan,
        'lagMedian1': np.nan,
        
    }, index=df.index
    ))
    
    ## computing the lag variables
    lagFeatures = ["lagMedian12", "lagMean12","lagMedianFirst3","lagMeanFirst3",
                     "lagMedian6","lagMean6","lagMedian4","lagMean4","lagMedian3",
                     "lagMean3","lagMean2","lagMedian1"]
   
    
    for name, group in newdf.groupby(['series', 'period']):
        #print(name[0], name[1])
        mask = (newdf['series'] == name[0]) & (newdf['period'] == name[1])
        newdf.loc[mask, 'lagMedian12'] = newdf.loc[mask,'qtty'].median()
        newdf.loc[mask, 'lagMean12'] = newdf.loc[mask,'qtty'].mean()
        newdf.loc[mask, 'lagMedianFirst3'] = newdf.loc[mask,'qtty'].head(3).median()
        newdf.loc[mask, 'lagMeanFirst3'] = newdf.loc[mask,'qtty'].head(3).mean()
        newdf.loc[mask, 'lagMedian6'] = newdf.loc[mask,'qtty'].tail(6).median()
        newdf.loc[mask, 'lagMean6'] = newdf.loc[mask,'qtty'].tail(6).mean()
        newdf.loc[mask, 'lagMedian4'] = newdf.loc[mask,'qtty'].tail(4).median()
        newdf.loc[mask, 'lagMean4'] = newdf.loc[mask,'qtty'].tail(4).mean()
        newdf.loc[mask, 'lagMedian3'] = newdf.loc[mask,'qtty'].tail(3).median()
        newdf.loc[mask, 'lagMean3'] = newdf.loc[mask,'qtty'].tail(3).mean()
        newdf.loc[mask, 'lagMean2'] = newdf.loc[mask,'qtty'].tail(2).mean()
        newdf.loc[mask, 'lagMedian1'] = newdf.loc[mask,'qtty'].tail(1).median()
         
    
    newdf.period = newdf.period - 1
    
    df = pd.merge(newdf0, newdf[['series','period'] + lagFeatures], 
                  on= ['series','period'] )
    df = df.drop_duplicates(['series','iMonth','period'])
    ## randomly sample 20% of the last year data as validation set
    df = df.dropna(axis = 0, how = 'any')
    val_rows = np.random.choice(df[df.period==3].index.values, math.floor(df.index[df.period==3].shape[0]*0.2))
    df.mark[val_rows] = 1
    
    df = df.sort_values(by=['series','iMonth'])
    return df
    

file_path = '/Users/zhangrui/Downloads/M3.csv'

df = processing_raw_file(file_path)


category_feature_list = ['series', 'cate', 'year', 'month']
label_enc_list = []

for category_feature in category_feature_list:
    label_enc = preprocessing.LabelEncoder()
    label_enc.fit(df.loc[:,category_feature])
    label_enc_list.append(label_enc)
    df.loc[:, category_feature] = label_enc.transform(df.loc[:,category_feature])


"""
Normalization of numeric data
"""
numeric_feature_list = ["iMonth","lagMedian12","lagMean12",
                "lagMedianFirst3","lagMedianFirst3",
                "lagMedian6","lagMean6",
                "lagMedian4","lagMean4",
                "lagMedian3","lagMean3",
                "lagMean2",  "lagMedian1"]

df[numeric_feature_list] = preprocessing.scale(df[numeric_feature_list])
with open('modelPrepare.pkl', 'wb') as f:
        pickle.dump([df, label_enc_list], f, -1)



