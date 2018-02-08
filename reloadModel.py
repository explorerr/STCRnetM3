import re,os,sys
import operator
import math
import fnmatch

import _pickle as pickle

import numpy as np
import pandas as pd
from sklearn import preprocessing

import mxnet as mx
from mxnet import nd, gluon, autograd
from mxnet.gluon import nn, rnn

###The self define models
from DLHelper import DLPred,nnTrain, save_checkpoint, smape
from DLModels import  resnetSP

context = mx.gpu(1)
"""
Load the data with pickle
"""
with open('modelPrepare.pkl', 'rb') as f:
    dt, label_enc_list = pickle.load(f)

sub_test  = dt[dt.mark==2]

featureList = ['Series','Category', 'year','month',"iMonth", "lagMedian12", "lagMedian6", "lagMedian3", "lagMedian1",
                        "lagMean12", "lagMean6", "lagMean3"]
test_X = nd.array(sub_test.loc[:, featureList], ctx=context)

"""
Reload the model
"""
trainName = 'Resnet'
ResnetV5Dict ={}
for file in os.listdir('checkpoints/'):
    if fnmatch.fnmatch(file, 'mark_'+trainName+'huber'+'*.param'):
        print(file)
        try:
            metrics = float(re.search('_metrics_(.+?).param', file).group(1))
        except AttributeError:
            print("There is an AttributeError")
        ResnetV5Dict[file]=metrics

totalPredList = []
subModel = resnetSP(activation='relu',residualVariants=5)
subDict = ResnetV5Dict
sortedSubDict = sorted(subDict.items(), key=operator.itemgetter(1))
subPredList = []

sortedSubDict = sortedSubDict[0:10]
for j in range(len(sortedSubDict)):
    subFileName = 'checkpoints/'+sortedSubDict[j][0]
    subModel.load_params(subFileName, ctx=context)
    subPred = DLPred(subModel, test_X)[:,0].asnumpy()
    subPredList.append(subPred)
subPredData = np.transpose(np.vstack(subPredList))
subNNPred = subPredData[:,0]
subNNBagging = np.mean(subPredData, axis=1)
totalPredList.append(np.column_stack([subNNPred,subNNBagging]))

totalPredData = np.column_stack(totalPredList)
totalPredData = pd.DataFrame(data=totalPredData[:,:], columns=[ "resnetV5", "resnetV5Bagging"])

totalData = sub_test[['Series','Category', 'year','month',"qtty"]]
totalData = pd.concat([totalData.reset_index(drop=True), totalPredData.reset_index(drop=True)], axis=1)
print(smape(np.array(totalData['qtty']), np.array(totalData['resnetV5'])))
print(smape(np.array(totalData['qtty']), np.array(totalData['resnetV5Bagging'])))

