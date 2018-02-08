import os
import math
import _pickle as pickle

import numpy as np
import pandas as pd
from sklearn import preprocessing

dt = pd.read_csv("modelData.csv", na_values=['NA', 'NULL', 'NaN', '\\N'])

"""
label encoding of category features
"""
category_feature_list = ["Series","Category","year", "month"]
label_enc_list = []
for category_feature in category_feature_list:
    label_enc = preprocessing.LabelEncoder()
    label_enc.fit(dt.loc[:, category_feature])
    label_enc_list.append(label_enc)
    dt.loc[:, category_feature] = label_enc.transform(dt.loc[:, category_feature])

"""
Normalization of numeric data
"""
numeric_feature_lsit = ["iMonth","lagMedian12","lagMean12",
                "lagMedianFirst3","lagMedianFirst3",
                "lagMedian6","lagMean6",
                "lagMedian4","lagMean4",
                "lagMedian3","lagMean3",
                "lagMean2","lagMedian1"]
dt[numeric_feature_lsit] = preprocessing.scale(dt[numeric_feature_lsit])
with open('modelPrepare.pkl', 'wb') as f:
        pickle.dump([dt, label_enc_list], f, -1)

