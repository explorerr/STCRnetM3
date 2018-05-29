#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 14:23:37 2018

@author: zhangrui
"""

import torch
import torch.nn as nn


class Residual5(torch.nn.Module):

    def __init__(self, xDim, H):
        """
        In this class we define a residual block
        """
        super(Residual5, self).__init__()
        self.bn1 = nn.BatchNorm2d(xDim)
        self.fc1 = nn.Linear(xDim, H)
        self.bn2 = nn.BatchNorm2d(H)
        self.fc2 = nn.Linear(H, xDim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        out = self.bn1(x)
        out = self.relu(out)
        out = self.fc1(out).clamp(min=0)
        out = self.fc2(self.relu(self.bn2(out)))
        return (out + x)


class ResnetEB(torch.nn.Sequential):

    def __init__(self, xDim, H):

        super(ResnetEB, self).__init__()
        #self.net = torch.nn.Sequential()
        #self.outputLayer = torch.nn.Sequential()
        self.id_embedding = nn.Embedding(1428, 10)
        self.type_embedding = nn.Embedding(6, 2)
        self.nYear_embedding = nn.Embedding(149, 2)
        self.nMonth_embedding = nn.Embedding(12, 2)
        # self.net = torch.nn.Sequential(
        #     nn.Linear(22, H),
        #     nn.ReLU(True),
        #     torch.nn.Linear(H, 1)
        # )
        xDim = xDim + 9 + 1 + 1 + 1
        self.net = torch.nn.Sequential(Residual5(xDim, H))
        self.outputLayer = torch.nn.Sequential(
            nn.Linear(xDim, H),
            nn.BatchNorm2d(H),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(H, 1))

    def forward(self, x_f, x_l):
        embed_concat = torch.cat(
            (self.id_embedding(x_l[:, 0]),
             self.type_embedding(x_l[:, 1]),
             self.nYear_embedding(x_l[:, 2]),
             self.nMonth_embedding(x_l[:, 3]),
             x_f[:, 0:6]),
            1)
        out = self.net(embed_concat)
        #print(out.size())
        return self.outputLayer(out)


#word_to_ix = {"hello": 0, "world": 1}
# embeds = torch.nn.Embedding(2, 5)  # 2 words in vocab, 5 dimensional embeddings
#lookup_tensor = torch.LongTensor([word_to_ix["hello"], word_to_ix["world"]])
#hello_embed = embeds(torch.autograd.Variable(lookup_tensor))
# print(hello_embed)
#
#
# embeds = nn.Embedding(1428, 10)  # 2 words in vocab, 5 dimensional embeddings
#hello_embed = embeds(sub_train_X_l[:,0])
# print(hello_embed)
#


"""
A fully-connected ReLU network with one hidden layer, trained to predict y from x
by minimizing squared Euclidean distance.
This implementation defines the model as a custom Module subclass. Whenever you
want a model more complex than a simple sequence of existing Modules you will
need to define your model this way.
"""


class TwoLayerNet(torch.nn.Module):

    def __init__(self, D_in, H, D_out):

        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


class RNN(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )
        self.out = nn.Linear(64, 10)

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)  # x: (batch, time_step, input_size)
        out = self.out(r_out[:, -1, :])  # (batch, time_step, input)
        return out
