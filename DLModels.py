import mxnet as mx
from mxnet import gluon
from mxnet import nd
from mxnet.gluon import nn

class Residual5(nn.HybridBlock):
    def __init__(self, xDim,  **kwargs):
        super(Residual5, self).__init__(**kwargs)
        self.fc1 = nn.Dense(16)
        self.bn1 = nn.BatchNorm()
        self.fc2 = nn.Dense(units=xDim)
        self.bn2 = nn.BatchNorm()

    def hybrid_forward(self,F, x):
        out = self.fc1(nd.relu(self.bn1(x)))
        out = self.fc2(nd.relu(self.bn2(out)))
 #       return nd.relu(out + x)
        return (out + x)

class resnetSP(nn.HybridBlock):
    def __init__(self,activation='relu',residualVariants=1, **kwargs):
        # the activation test
        assert activation in set(['relu', 'softrelu', 'sigmoid','tanh'])
        super(resnetSP, self).__init__(**kwargs)
        self.net = nn.HybridSequential()
        self.outputLayer = nn.HybridSequential()
        with self.name_scope():
            self.store_embedding = nn.Embedding(1428,10)
            self.type_embedding = nn.Embedding(6,2)
            self.nYear_embedding = nn.Embedding(4,2)
            self.nMonth_embedding = nn.Embedding(12,2)
            self.net.add(Residual5(xDim=22))
            self.outputLayer.add(nn.Dense(64))
            self.outputLayer.add(nn.BatchNorm())
            self.outputLayer.add(nn.Activation(activation=activation))
            self.outputLayer.add(nn.Dropout(.2))
            self.outputLayer.add(nn.Dense(1, activation='relu'))

    def hybrid_forward(self,F, x):
        embed_concat = nd.concat(
                self.store_embedding(x[:,0]),
                self.type_embedding(x[:,1]),
                self.nYear_embedding(x[:,2]),
                self.nMonth_embedding(x[:,3]), x[:,4:10])
        return self.outputLayer(self.net(embed_concat))
