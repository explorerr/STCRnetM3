import os
import random
from os import path
import numpy as np
import mxnet as mx
from mxnet import gluon, nd, gpu, autograd
from mxnet.gluon import nn, rnn

abs_loss = gluon.loss.L1Loss()
context = mx.cpu(0)


def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred))
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return 200 * np.mean(diff)


def DLPred(net, dt):
    if(dt.shape[0] <= 60000):
        return net(dt)
    block_size = dt.shape[0] // 60000 + 1
    pred_result = net(dt[0:60000, ])
    for i in range(1, block_size):
        i = i * 60000
        j = min(i + 60000, dt.shape[0])
        block_pred = net(dt[i:j, ])
        pred_result = nd.concat(pred_result, block_pred, dim=0)
    return pred_result


def baggingPred(net, modelList, sub_valid_X):
    predList = []
    for param in modelList:
        param = "checkpoints/" + param
        net.load_params(param, ctx=context)
        tmpPred = DLPred(net, sub_valid_X)[:, 0].asnumpy()
        predList.append(tmpPred)
    predList = np.column_stack(predList)
    return np.mean(predList, axis=1)


def save_checkpoint(net, mark, valid_metric, save_path):
    if not path.exists(save_path):
        os.makedirs(save_path)
    filename = path.join(save_path, "mark_{:s}_metrics_{:.3f}".format(mark, valid_metric))
    filename += '.param'
    net.save_params(filename)


def nnTrain(model_mark, nnModel, train_data, valid_data_X, valid_data_Y, test_data_X, test_data_Y, batch_size, loss_func, epochs,
            optimizer, optimizer_params, lr_decay_rate=1):
    """
    Providing 3 approaches to train the model: momentum, adadelta and adam
    """
    assert optimizer in set(['sgd', 'adadelta', 'adam'])
    random.seed(1)
    train_iter = gluon.data.DataLoader(train_data, batch_size, shuffle=True)
    nTrain = len(train_data)
    nValid = len(test_data_Y)
    # The model
    mx.random.seed(123456)
    nnModel.collect_params().initialize(mx.initializer.MSRAPrelu(), ctx=context)
    trainer = gluon.Trainer(nnModel.collect_params(), optimizer=optimizer, optimizer_params=optimizer_params)

    best_smape = 1
    for e in range(epochs):
     #       if(e>=2): trainer.set_learning_rate(trainer.learning_rate * lr_decay_rate)
        train_loss = 0
        for data, label in train_iter:
            label = label.as_in_context(context)
            with autograd.record():
                output = nnModel(data)
                loss = loss_func(output, label)
            loss.backward()
            trainer.step(batch_size)
            train_loss += nd.sum(loss).asscalar()
        # The valid loss
        valid_pred = DLPred(nnModel, valid_data_X)[:, 0].asnumpy()
        valid_true = valid_data_Y.asnumpy()
        # The valid loss
        test_pred = DLPred(nnModel, test_data_X)[:, 0].asnumpy()
        test_true = test_data_Y.asnumpy()

        valid_loss = nd.sum(abs_loss(nd.array(valid_true), nd.array(valid_pred))).asscalar()
        test_loss = nd.sum(abs_loss(nd.array(test_true), nd.array(test_pred))).asscalar()

        valid_smape = smape(valid_true, valid_pred)
        test_smape = smape(test_true, test_pred)

        print("Epoch %d, train loss: %f, valid_loss: %f" % (e, train_loss / nTrain, valid_loss / nValid))
        print("Valid smape  %f; Test smape %f" % (valid_smape, test_smape))
        # Save the model
        if(e == 0 or valid_smape < best_smape):
            best_smape = valid_smape
        if e > 0 and valid_smape < best_smape + 0.3:
            save_checkpoint(nnModel, model_mark + str(e), round(valid_smape, 2), save_path="checkpoints")
