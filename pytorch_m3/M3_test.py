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
import os
import argparse
import json
import pandas as pd


GPU_DEV = -1
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
checkpoint_dir = ''

FLOATTYPE = torch.FloatTensor
LONGTYPE = torch.LongTensor
float_featureList = []
long_featureList = []
input_file_name = ''
config_data = ''

error_metric_name = ''

resume_from = None


def read_config(args):

    global BATCH_SIZE, LR, NUM_EPOCH, SEQUENCE_LEN, GPU_DEV
    global print_every, checkpoint_every, checkpoint_name, checkpoint_dir
    global GRAD_CLIP, LR_DECAY_EVERY, LR_DECAY_FACTOR, OPTIMIZER
    global MODEL_NAME, CRITERION
    global input_file_name, float_featureList, long_featureList, config_data, resume_from, error_metric_name

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
    checkpoint_dir = config_data['Common']['checkpoint_dir']
    error_metric_name = config_data['Common']['error_metric']
    GPU_DEV = config_data['Common']['gpu']
    if not isinstance(GPU_DEV, int):
        raise ValueError("gpu device not an integer: ", GPU_DEV)
    if GPU_DEV > torch.cuda.device_count():
        raise ValueError("gpu device number ({}) larger than available gpu device number ({})".format(GPU_DEV, torch.cuda.device_count()))

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
    try:
        resume_from = config_data['Common']['resume_from']
    except:
        pass


def load_data(device):

    # load the data:
    with open(input_file_name, 'rb') as f:
        dt, label_enc_list = pickle.load(f)

    sub_train = dt[dt.mark == 0]
    sub_valid = dt[dt.mark == 1]
    sub_test = dt[dt.mark == 2]

    # ['series', 'n', 'nf', 'cate', 'year', 'month', 'iMonth', 'qtty', 'mark', 'period', 'lagMedian12', 'lagMean12', 'lagMedianFirst3', 'lagMeanFirst3', 'lagMedian6', 'lagMean6', 'lagMedian4', 'lagMean4', 'lagMedian3', 'lagMean3', 'lagMean2', 'lagMedian1'], dtype='object')

    if len(float_featureList) > 0:
        sub_train_X_f = Variable(torch.from_numpy(sub_train[float_featureList].values).type(FLOATTYPE))
        sub_valid_X_f = Variable(torch.from_numpy(sub_valid[float_featureList].values).type(FLOATTYPE))
        sub_test_X_f = Variable(torch.from_numpy(sub_test[float_featureList].values).type(FLOATTYPE))

    if len(long_featureList) > 0:
        sub_train_X_l = Variable(torch.from_numpy(sub_train[long_featureList].values).type(LONGTYPE))
        sub_valid_X_l = Variable(torch.from_numpy(sub_valid[long_featureList].values).type(LONGTYPE))
        sub_test_X_l = Variable(torch.from_numpy(sub_test[long_featureList].values).type(LONGTYPE))

    sub_train_Y = Variable(torch.from_numpy(sub_train['qtty'].values).type(FLOATTYPE))
    sub_train_Y = sub_train_Y.view((sub_train_X_f.size(0), 1))
    # print('sub_train_Y size: ', sub_train_Y.size())
    sub_valid_Y = Variable(torch.from_numpy(sub_valid['qtty'].values).type(FLOATTYPE))
    sub_test_Y = Variable(torch.from_numpy(sub_test['qtty'].values).type(FLOATTYPE))
    # print(sub_train_Y.size())
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


def get_err(data, y_pred, batch_y, model, epoch, epoch_start, step, loss, device):
    y_pred_val = model(data['sub_valid_X_f'].to(device=device), data['sub_valid_X_l'].to(device=device)).cpu()
    y_pred_test = model(data['sub_test_X_f'].to(device=device), data['sub_test_X_l'].to(device=device)).cpu()
    cur = pd.DataFrame({'epoch': [epoch + epoch_start], 'step': [step], "loss": [loss.item()],
                        'training_' + error_metric_name:
                        [error_metric(y_pred[:, 0], batch_y[:, 0], error_metric_name)],
                        'validation_' + error_metric_name:
                        [error_metric(y_pred_val[:, 0], data['sub_valid_Y'], error_metric_name)],
                        'testing_' + error_metric_name:
                        [error_metric(y_pred_test[:, 0], data['sub_test_Y'], error_metric_name)]
                        })
    return (y_pred_val, y_pred_test, cur)


def main(argv=None):

    global BATCH_SIZE, LR, NUM_EPOCH, error_metric_name
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', help="the configuration file with input specifications", default='config.json')
    args = parser.parse_args()

    read_config(args)

    if GPU_DEV == -1:
        device = torch.device("cpu")
    elif GPU_DEV >= 0:
        device = torch.device("cuda:{}".format(GPU_DEV))

    data = load_data(device)

    history = pd.DataFrame()

    # make sure checkpoint saving dir exist
    if not os.path.isdir(os.path.join(os.getcwd(), checkpoint_dir)):
        os.mkdir(os.path.join(os.getcwd(), checkpoint_dir))

    # model = TwoLayerNet(sub_train_X.shape[1], 20, 1)
    # model = ResnetEB(100)
    xDim = data['sub_train_X_f'].size()[1] + data['sub_train_X_l'].size()[1]
    # print('xDim=', xDim)
    print('input dimension: ', data['sub_train_X_f'].size()[0], xDim, )
    if MODEL_NAME == "ResnetEB":
        model = ResnetEB(xDim, config_data['Model']['NN']['HIDDEN'])
        if resume_from is None:
            model.apply(weights_init)
            epoch_start = 0
        else:
            resume_from_file = checkpoint_dir + checkpoint_name + resume_from['file'] + '.pkl'
            epoch_start = resume_from['epoch']
            if not os.path.isfile(resume_from_file):
                raise Exception("resume_from file can not be found. resume_from:", resume_from_file)
            model.load_state_dict(torch.load(resume_from_file))

    model = model.to(device=device)

    if CRITERION == "MSE":
        criterion = torch.nn.MSELoss(size_average=False)
    elif CRITERION == "ABS":
        criterion = torch.nn.L1Loss()
    if OPTIMIZER == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    if OPTIMIZER == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    num_train = data['sub_train_X_f'].size()[0]
    if BATCH_SIZE == 0:
        BATCH_SIZE = num_train
    # print(num_train, BATCH_SIZE, num_train / BATCH_SIZE - 1)
    # print(num_train, BATCH_SIZE)
    for epoch in range(NUM_EPOCH):
        # permutate the indecies for mini-batch training
        if BATCH_SIZE < num_train:
            permutation = torch.randperm(num_train)

        for step in range(0, num_train, BATCH_SIZE):
            # print( "step ", step)
            optimizer.zero_grad()
            if BATCH_SIZE < num_train:
                indices = permutation[step: step + BATCH_SIZE]
                batch_x_f, batch_x_l, batch_y = data['sub_train_X_f'][indices], data['sub_train_X_l'][indices], data['sub_train_Y'][indices]
            else:
                batch_x_f, batch_x_l, batch_y = data['sub_train_X_f'], data['sub_train_X_l'], data['sub_train_Y']

            batch_x_f = batch_x_f.to(device=device)
            batch_x_l = batch_x_l.to(device=device)
            batch_y = batch_y.to(device=device)
            # print(batch_x_f.size())
            y_pred = model(batch_x_f, batch_x_l)
            loss = criterion(y_pred, batch_y)
            # print(loss)

            if step == range(0, num_train, BATCH_SIZE)[-1] and epoch % print_every == 0:

                y_pred_val, y_pred_test, cur = get_err(data, y_pred, batch_y, model, epoch, epoch_start, step, loss, device)

                print('Epoch[{}]-Step[{}]'.format(epoch + epoch_start, step),
                      ': loss = {}'.format(loss.item()),
                      '  |  training {} = {:.4f}'.format(error_metric_name,
                                                         cur['training_' + error_metric_name][0]),
                      '  |  validation {} = {:.4f}'.format(error_metric_name,
                                                           cur['validation_' + error_metric_name][0]),
                      '  |  testing {} = {:.4f}'.format(error_metric_name,
                                                        cur['testing_' + error_metric_name][0]))
                history = history.append(cur, sort=True)

            if step == range(0, num_train, BATCH_SIZE)[-1] and epoch % checkpoint_every == 0:

                y_pred_val, y_pred_test, cur = get_err(data, y_pred, batch_y, model, epoch, epoch_start, step, loss, device)

                history = history.append(cur, sort=True)

                check = {"opt": config_data, "history": history.reset_index().to_json()}
                # print(y_pred_val.detach().numpy().shape)
                y_val = pd.DataFrame({"y_val": data['sub_valid_Y'], 'y_val_pred': y_pred_val.detach().numpy()[:, 0]})
                y_test = pd.DataFrame({"y_val": data['sub_test_Y'], 'y_val_pred': y_pred_test.detach().numpy()[:, 0]})

                with open('{}{}_{}_{}.json'.format(checkpoint_dir, checkpoint_name, epoch + epoch_start, step), 'w') as fp:
                    json.dump(check, fp)
                # torch.save(model, '{}_{}_{}.pkl'.format(checkpoint_name, epoch, step))  # entire net
                torch.save(model.state_dict(), '{}{}_param_{}_{}.pkl'.format(checkpoint_dir, checkpoint_name, epoch + epoch_start, step))  # parameters

                # print(type(y_pred_val.data.numpy()),
                #       y_pred_val.data.numpy().shape,
                #       y_pred_val.data.numpy()[:, 0])

                # print(type(data['sub_valid_Y'].data.numpy()),
                #       data['sub_valid_Y'].data.numpy().shape,
                #       data['sub_valid_Y'].data.numpy())

                y_val = pd.DataFrame({"y_val": data['sub_valid_Y'].data.numpy(),
                                      'y_val_pred': y_pred_val.data.numpy()[:, 0]})
                y_test = pd.DataFrame({"y_test": data['sub_test_Y'].data.numpy(),
                                       'y_test_pred': y_pred_test.data.numpy()[:, 0]})

                y_val.to_csv('{}{}_{}_{}_val.csv'.format(checkpoint_dir, checkpoint_name, epoch + epoch_start, step), index=False)
                y_test.to_csv('{}{}_{}_{}_test.csv'.format(checkpoint_dir, checkpoint_name, epoch + epoch_start, step), index=False)

            optimizer.zero_grad()   # clear gradients for next train
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()


if __name__ == "__main__":
    main()
