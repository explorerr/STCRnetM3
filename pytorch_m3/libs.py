

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
