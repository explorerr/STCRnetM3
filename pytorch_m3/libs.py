

import _pickle as pickle
import torch
from torch.autograd import Variable


def save(net, net_name):
    torch.save(net, net_name + '.pkl')  # entire net
    torch.save(net.state_dict(), net_name + '_params.pkl')  # parameters


def restore_net(net_name):
    return(torch.load(name))


def restore_params(net, et_name):

    return(net.load_state_dict(torch.load(net_name + '_params.pkl')))


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, mean=1)
        torch.nn.init.constant_(m.bias.data, 0)
    elif classname.find('Embedding') != -1:
        torch.nn.init.normal_(m.weight.data)
    elif classname.find('Linear') != -1:
        torch.nn.init.normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0)


def MAPE(y_hat, y):
    return(torch.mean(torch.abs((y_hat - y) / y)).item())


def MAE(y_hat, y):
    return(torch.mean(torch.abs((y_hat - y))).item())


def MBE(y_hat, y):
    return(torch.mean(y_hat - y).item())


def SMAPE(y_hat, y):
    #print("in SMAPE: y.shape=", y.shape, '   y_hat.shape=', y_hat.shape)
    return(torch.mean(torch.abs(y - y_hat) / (y + y_hat))).item()


def error_metric(y_hat, y, error_metric_name):
    # print(error_metric_name)
    if error_metric_name == 'SMAPE':
        return(SMAPE(y_hat, y))
    if error_metric_name == 'MAPE':
        return(MAPE(y_hat, y))
