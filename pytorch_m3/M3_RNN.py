#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 10:01:48 2018

@author: zhangrui
"""
import _pickle as pickle
import torch
from torch.autograd import Variable
from models import ResnetEB, TwoLayerNet

floatdtype = torch.FloatTensor
longdtype = torch.LongTensor

# load the data:

with open('modelPrepare.pkl', 'rb') as f:
    dt, label_enc_list = pickle.load(f)

sub_train = dt[dt.mark == 0]
sub_valid = dt[dt.mark == 1]
sub_test = dt[dt.mark == 2]

float_featureList = ["iMonth", "lagMedian12", "lagMedian6", "lagMedian3",
                     "lagMedian1", "lagMean3"]


long_featureList = ['series', 'cate', 'year', 'month']

sub_train_X_f = Variable(torch.from_numpy(sub_train.as_matrix(columns=float_featureList)).type(floatdtype))
sub_train_X_l = Variable(torch.from_numpy(sub_train.as_matrix(columns=long_featureList)).type(longdtype))
sub_train_Y = Variable(torch.from_numpy(sub_train.as_matrix(columns=['qtty'])).type(floatdtype))
sub_valid_X_f = Variable(torch.from_numpy(sub_valid.as_matrix(columns=float_featureList)).type(floatdtype))
sub_valid_X_l = Variable(torch.from_numpy(sub_valid.as_matrix(columns=long_featureList)).type(longdtype))
sub_valid_Y = Variable(torch.from_numpy(sub_valid.as_matrix(columns=['qtty'])).type(floatdtype))
sub_test_X_f = Variable(torch.from_numpy(sub_test.as_matrix(columns=float_featureList)).type(floatdtype))
sub_test_X_l = Variable(torch.from_numpy(sub_test.as_matrix(columns=long_featureList)).type(longdtype))
sub_test_Y = Variable(torch.from_numpy(sub_test.as_matrix(columns=['qtty'])).type(floatdtype))


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight.data)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal(m.weight.data, mean=1)
        torch.nn.init.constant(m.bias.data, 0)
    elif classname.find('Embedding') != -1:
        torch.nn.init.normal(m.weight.data)
    elif classname.find('Linear') != -1:
        torch.nn.init.normal(m.weight.data)
        torch.nn.init.constant(m.bias.data, 0)


# model = TwoLayerNet(sub_train_X.shape[1], 20, 1)
model = ResnetEB(100)
model.apply(weights_init)
criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=2e-12)

for t in range(3000):

    y_pred = model(sub_train_X_f, sub_train_X_l)
    loss = criterion(y_pred, sub_train_Y)
    print('Step[{}]: loss = {}      MAPE = {:1.4f}'.format(t, loss.data[0],
                                                      torch.mean(torch.abs((y_pred - sub_train_Y) / sub_train_Y)).data[0]))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
