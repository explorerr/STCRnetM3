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
from libs import *

import argparse
import json
import pandas as pd


BATCH_SIZE = 500
LR = 1e-12
LR_DECAY_FACTOR = .5
LR_DECAY_EVERY = 5
NUM_EPOCH = 5000
MODEL_NAME = 'ResnetEB'
CRITERION = "MSE"
print_every = 100
checkpoint_every = 100
checkpoint_name = 'checkpoint'

FLOATTYPE = torch.FloatTensor
LONGTYPE = torch.LongTensor
float_featureList = []
long_featureList = []
input_file_name = ''
config_data = ''


def MAPE(y_hat, y):
    return(torch.mean(torch.abs((y_hat - y) / y)).data[0])


def MAE(y_hat, y):
    return(torch.mean(torch.abs((y_hat - y))).data[0])


def MBE(y_hat, y):
    return(torch.mean(y_hat - y).data[0])


def read_config(args):

    global BATCH_SIZE, LR, NUM_EPOCH, SEQUENCE_LEN
    global print_every, checkpoint_every, checkpoint_name
    global GRAD_CLIP, LR_DECAY_EVERY, LR_DECAY_FACTOR, OPTIMIZER
    global MODEL_NAME, CRITERION
    global input_file_name, float_featureList, long_featureList, config_data

    config_data = json.load(open(args.config_file))

    BATCH_SIZE = config_data['Training']['BATCH_SIZE']
    LR = config_data['Training']['LR']
    NUM_EPOCH = config_data['Training']['NUM_EPOCH']
    try:
        SEQUENCE_LEN = config_data['Training']['SEQUENCE_LEN']
    except:
        pass
    try:
        GRAD_CLIP = config_data['Training']['GRAD_CLIP']
    except:
        pass
    try:
        LR_DECAY_EVERY = config_data['Training']['LR_DECAY_EVERY']
    except:
        pass
    try:
        LR_DECAY_FACTOR = config_data['Training']['LR_DECAY_FACTOR']
    except:
        pass

    OPTIMIZER = config_data['Training']['OPTIMIZER']
    CRITERION = config_data['Training']['CRITERION']
    print_every = config_data['Common']['print_every']
    checkpoint_every = config_data['Common']['checkpoint_every']
    checkpoint_name = config_data['Common']['checkpoint_name']

    MODEL_NAME = config_data['Model']['MODEL_NAME']

    input_file_name = config_data['Dataset']['file_name']
    try:
        float_featureList = config_data['Dataset']['real_value_features']
    except:
        pass
    try:
        long_featureList = config_data['Dataset']['categorical_features']
    except:
        pass


def load_data():

    # load the data:
    with open(input_file_name, 'rb') as f:
        dt, label_enc_list = pickle.load(f)

    sub_train = dt[dt.mark == 0]
    sub_valid = dt[dt.mark == 1]
    sub_test = dt[dt.mark == 2]

    # ['series', 'n', 'nf', 'cate', 'year', 'month', 'iMonth', 'qtty', 'mark', 'period', 'lagMedian12', 'lagMean12', 'lagMedianFirst3', 'lagMeanFirst3', 'lagMedian6', 'lagMean6', 'lagMedian4', 'lagMean4', 'lagMedian3', 'lagMean3', 'lagMean2', 'lagMedian1'], dtype='object')

    if len(float_featureList) > 0:
        sub_train_X_f = Variable(torch.from_numpy(sub_train.as_matrix(columns=float_featureList)).type(FLOATTYPE))
        sub_valid_X_f = Variable(torch.from_numpy(sub_valid.as_matrix(columns=float_featureList)).type(FLOATTYPE))
        sub_test_X_f = Variable(torch.from_numpy(sub_test.as_matrix(columns=float_featureList)).type(FLOATTYPE))

    if len(long_featureList) > 0:
        sub_train_X_l = Variable(torch.from_numpy(sub_train.as_matrix(columns=long_featureList)).type(LONGTYPE))
        sub_valid_X_l = Variable(torch.from_numpy(sub_valid.as_matrix(columns=long_featureList)).type(LONGTYPE))
        sub_test_X_l = Variable(torch.from_numpy(sub_test.as_matrix(columns=long_featureList)).type(LONGTYPE))

    sub_train_Y = Variable(torch.from_numpy(sub_train.as_matrix(columns=['qtty'])).type(FLOATTYPE))
    sub_valid_Y = Variable(torch.from_numpy(sub_valid.as_matrix(columns=['qtty'])).type(FLOATTYPE))
    sub_test_Y = Variable(torch.from_numpy(sub_test.as_matrix(columns=['qtty'])).type(FLOATTYPE))

    return({"sub_train_X_f": sub_train_X_f,
            "sub_valid_X_f": sub_valid_X_f,
            "sub_test_X_f": sub_test_X_f,
            "sub_train_X_l": sub_train_X_l,
            "sub_valid_X_l": sub_valid_X_l,
            "sub_test_X_l": sub_test_X_l,
            "sub_train_Y": sub_train_Y,
            "sub_valid_Y": sub_valid_Y,
            "sub_test_Y": sub_test_Y
            })


def main(argv=None):

    global BATCH_SIZE, LR, NUM_EPOCH
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', help="the configuration file with input specifications", default='wea_config.json')
    args = parser.parse_args()

    read_config(args)

    data = load_data()

    history = pd.DataFrame()

    # model = TwoLayerNet(sub_train_X.shape[1], 20, 1)
    # model = ResnetEB(100)
    if MODEL_NAME == "ResnetEB":
        model = ResnetEB(config_data['Model']['NN']['HIDDEN'])
        model.apply(weights_init)
    if CRITERION == "MSE":
        criterion = torch.nn.MSELoss(size_average=False)
    if OPTIMIZER == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=LR)

    num_train = data['sub_train_X_f'].size()[0]
    # print(num_train)
    for epoch in range(NUM_EPOCH):
        # permutate the indecies for mini-batch training
        permutation = torch.randperm(num_train)

        for step in range(0, num_train, BATCH_SIZE):

            optimizer.zero_grad()
            indices = permutation[step: step + BATCH_SIZE]
            batch_x_f, batch_x_l, batch_y = data['sub_train_X_f'][indices], data['sub_train_X_l'][indices], data['sub_train_Y'][indices]
            y_pred = model(batch_x_f, batch_x_l)
            loss = criterion(y_pred, batch_y)

            if step > 0 and step % print_every == 0:

                y_pred_val = model(data['sub_valid_X_f'], data['sub_valid_X_l'])
                y_pred_test = model(data['sub_test_X_f'], data['sub_test_X_l'])
                cur = pd.DataFrame({'epoch': [epoch], 'step': [step], "loss": [loss.data[0]],
                                    'training_mape': [MAPE(y_pred, batch_y)],
                                    'validation_mape': [MAPE(y_pred_val, data['sub_valid_Y'])],
                                    'testing_mape': [MAPE(y_pred_test, data['sub_test_Y'])]
                                    })
                print('Epoch[{}]-Step[{}]'.format(epoch, step),
                      ': loss = {}'.format(loss.data[0]),
                      '  |  training MAPE = {0:.4f}'.format(cur.training_mape[0]),
                      '  |  validation MAPE = {:.4f}'.format(cur.validation_mape[0]),
                      '  |  testing MAPE = {:.4f}'.format(cur.testing_mape[0]))
                history = history.append(cur)

            if step > 0 and step % checkpoint_every == 0:
                y_pred_val = model(data['sub_valid_X_f'], data['sub_valid_X_l'])
                y_pred_test = model(data['sub_test_X_f'], data['sub_test_X_l'])
                cur = pd.DataFrame({'epoch': [epoch], 'step': [step], "loss": [loss.data[0]],
                                    'training_mape': [MAPE(y_pred, batch_y)],
                                    'validation_mape': [MAPE(y_pred_val, data['sub_valid_Y'])],
                                    'testing_mape': [MAPE(y_pred_test, data['sub_test_Y'])]
                                    })
                history = history.append(cur)

                check = {"opt": config_data, "history": history.reset_index().to_json()}

                with open('{}_{}_{}.json'.format(checkpoint_name, epoch, step), 'w') as fp:
                    json.dump(check, fp)
                torch.save(model, '{}_{}_{}.pkl'.format(checkpoint_name, epoch, step))  # entire net
                torch.save(model.state_dict(), '{}_param_{}_{}.pkl'.format(checkpoint_name, epoch, step))  # parameters

            optimizer.zero_grad()   # clear gradients for next train
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()


if __name__ == "__main__":
    main()
