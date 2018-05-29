import sys
import math
import _pickle as pickle

import numpy as np
import pandas as pd
from sklearn import preprocessing

import mxnet as mx
from mxnet import nd, gluon, autograd
from mxnet.gluon import nn, rnn

# The self define models
from DLHelper import DLPred, nnTrain, save_checkpoint
from DLModels import resnetSP

"""
The parameters parse
"""
#gpus = mx.test_utils.list_gpus()
context = mx.cpu(0) # [mx.gpu()] if gpus else [mx.cpu(0), mx.cpu(1)]
"""
Load the data with pickle
"""
with open('modelPrepare.pkl', 'rb') as f:
    dt, label_enc_list = pickle.load(f)

#monthId = 1
sub_train = dt[dt.mark == 0]
sub_valid = dt[dt.mark == 1]
sub_test = dt[dt.mark == 2]

featureList = ['Series', 'Category', 'year', 'month', "iMonth", "lagMedian12", "lagMedian6", "lagMedian3", "lagMedian1", "lagMean3"]
sub_train_X, sub_train_Y = nd.array(sub_train.loc[:, featureList], ctx=context), nd.array(sub_train.loc[:, 'qtty'], ctx=context)
sub_valid_X, sub_valid_Y = nd.array(sub_valid.loc[:, featureList], ctx=context), nd.array(sub_valid.loc[:, 'qtty'], ctx=context)
sub_test_X, sub_test_Y = nd.array(sub_test.loc[:, featureList], ctx=context), nd.array(sub_test.loc[:, 'qtty'], ctx=context)

sub_train_nd = gluon.data.ArrayDataset(sub_train_X, sub_train_Y)
"""
The model training
"""
#huber_loss = gluon.loss.HuberLoss()
abs_loss = gluon.loss.L1Loss()
square_loss = gluon.loss.L2Loss()

trainName = 'Resnet'

# choose parameters
batch_size = 512
epoch_num = 200
optimizer = 'adam'
optimizer_params = {'learning_rate': 0.05}

model1 = resnetSP(activation='softrelu', residualVariants=5)
nnTrain(trainName + "huber", model1, sub_train_nd, sub_valid_X, sub_valid_Y, sub_test_X, sub_test_Y,
        batch_size=batch_size, loss_func=abs_loss, epochs=epoch_num, optimizer=optimizer, optimizer_params=optimizer_params, lr_decay_rate=1)
