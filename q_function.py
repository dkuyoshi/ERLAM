import chainer
import numpy as np
from chainer import cuda
from chainer import Chain, Sequential, ChainList, Variable
from chainer import links as L
from chainer import functions as F

from chainerrl.action_value import DiscreteActionValue
from chainerrl.q_function import StateQFunction
from chainerrl.recurrent import RecurrentChainMixin
from chainerrl.links import FactorizedNoisyLinear

from sklearn import random_projection


class DQNHead(Chain):
    def __init__(self, n_history=4, n_hidden=512):
        super().__init__()
        initializer = chainer.initializers.HeNormal()
        with self.init_scope():
            self.conv_layers = chainer.Sequential(
                L.Convolution2D(None, 32, 8, stride=4, initialW=initializer),
                F.relu,
                L.Convolution2D(None, 64, 4, stride=2, initialW=initializer),
                F.relu,
                L.Convolution2D(None, 64, 3, stride=1, initialW=initializer),
                F.relu)
            self.l1 = L.Linear(None, n_hidden, initialW=initializer)

    def __call__(self, x):
        h = self.conv_layers(x)
        h = self.l1(h)
        return h


class QFunction(Chain):
    def __init__(self, n_actions, n_history=4, n_hidden=512, dim=4):
        super().__init__()
        with self.init_scope():
            self.hout = DQNHead(n_history, n_hidden)
            self.qout = L.Linear(n_hidden, n_actions)

        # Random Projectionモデルの定義
        self.transformer = random_projection.SparseRandomProjection(n_components=dim, random_state=0)

    def __call__(self, x):
        self.l = self.hout(x)
        q = self.qout(F.relu(self.l))

        return DiscreteActionValue(q)

    def get_embedding(self, x):
        # Random Projection (1, 4)
        x_flat = x.flatten()
        x_new = self.transformer.fit_transform(x_flat.reshape(1, -1))
        return x_new

    def batch_get_embedding(self, x):
        embeddings = []
        for a in x:
            a_flat = a.flatten()
            a_new = self.transformer.fit_transform(a_flat.reshape(1, -1))
            embeddings.append(a_new)
        return self.xp.asarray(embeddings)

    def get_embedding_each_frame(self, x):
        # Random Projection (4, 1, 4) => (1, 16)の形
        embeddings = []
        x = cuda.to_cpu(x)
        for a_frame in np.squeeze(x):
            a_frame = a_frame.flatten()
            a_new = self.transformer.fit_transform(a_frame.reshape(1, -1))
            embeddings.append(a_new)
        # embeddings = cuda.to_gpu(embeddings)
        return np.concatenate(embeddings, axis=0).reshape(1, -1)

    def batch_get_embedding_each_frame(self, x):
        batch_embeddings = []
        x = cuda.to_cpu(x)
        for a_x in np.squeeze(x):
            embeddings = []
            for a_frame in a_x:
                a_flat = a_frame.flatten()
                a_new = self.transformer.fit_transform(a_flat.reshape(1, -1))
                embeddings.append(a_new.flatten())
            batch_embeddings.append(np.concatenate(embeddings, axis=0).reshape(1, -1))
        return np.asarray(batch_embeddings)


class DuelingQFunction(Chain):
    def __init__(self, n_actions, n_history=4, n_hidden=512, dim=4):
        super().__init__()
        with self.init_scope():
            self.hout = DQNHead(n_history, n_hidden)
            self.a_stream = L.Linear(n_hidden, n_actions)
            self.v_stream = L.Linear(n_hidden, 1)

        self.n_action = n_actions

        # Random Projectionモデルの定義
        self.transformer = random_projection.SparseRandomProjection(n_components=dim, random_state=0)

    def __call__(self, x):
        self.l = self.hout(x)
        activation = F.relu(self.l)

        batch_size = x.shape[0]
        ya = self.a_stream(activation)
        mean = F.reshape(F.sum(ya, axis=1) / self.n_action, (batch_size, 1))
        ya, mean = F.broadcast(ya, mean)
        ya -= mean

        ys = self.v_stream(activation)
        ya, ys = F.broadcast(ya, ys)
        q = ya + ys

        return DiscreteActionValue(q)

    def get_embedding(self, x):
        # Random Projection　(1, 4)の形
        x_flat = x.fltten()
        x_new = self.transformer.fit_transform(x_flat.reshape(1, -1))

        return x_new

    def batch_get_embedding(self, x):
        embeddings = []
        for a in x:
            a_flat = a.flatten()
            a_new = self.transformer.fit_transform(a_flat.reshape(1, -1))
            embeddings.append(a_new)
        return self.xp.asarray(embeddings)

    def get_embedding_each_frame(self, x):
        # Random Projection (4, 1, 4) => (1, 16)の形
        embeddings = []
        for a_frame in self.xp.squeeze(x):
            a_frame = a_frame.flatten()
            a_new = self.transformer.fit_transform(a_frame.reshape(1, -1))
            embeddings.append(a_new)

        return self.xp.concatenate(embeddings, axis=0).reshape(1, -1)

    def batch_get_embedding_each_frame(self, x):
        batch_embeddings = []
        for a_x in self.xp.squeeze(x):
            embeddings = []
            for a_frame in a_x:
                a_flat = a_frame.flatten()
                a_new = self.transformer.fit_transform(a_flat.reshape(1, -1))
                embeddings.append(a_new.flatten())
            batch_embeddings.append(self.xp.concatenate(embeddings, axis=0).reshape(1, -1))
        return self.xp.asarray(batch_embeddings)


class CartPoleHead(Chain):
    def __init__(self, obs_size, n_hidden=64, n_out=32):
        super().__init__()
        with self.init_scope():
            self.l0 = L.Linear(obs_size, n_hidden)
            self.l1 = L.Linear(n_hidden, n_out)

    def __call__(self, x, test=False):
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))
        return h


class QFunctionCartPole(Chain):
    def __init__(self, obs_size, n_actions, n_hidden=32, dim=4):
        super().__init__()
        with self.init_scope():
            self.hout = CartPoleHead(obs_size)
            self.qout = L.Linear(n_hidden, n_actions)
            
        self.transformer = random_projection.SparseRandomProjection(n_components=dim, random_state=0)

    def __call__(self, x):
        self.l = self.hout(x)
        q = self.qout(self.l)

        return DiscreteActionValue(q)

    def get_embedding(self, x):
        # Random Projection
        # x_flat = x.flatten()
        x_new = self.transformer.fit_transform(x.reshape(1, -1))

        return x_new

    def batch_get_embedding(self, x):
        embeddings = []
        for a in x:
            a_flat = a.flatten()
            a_new = self.transformer.fit_transform(a_flat.reshape(1, -1))
            embeddings.append(a_new)
        return self.xp.asarray(embeddings)

    def get_embedding_each_frame(self, x):
        # Random Projection
        # x_flat = x.flatten()
        x = cuda.to_cpu(x)
        x_new = self.transformer.fit_transform(x.reshape(1, -1))

        return x_new

    def batch_get_embedding_each_frame(self, x):
        embeddings = []
        x = cuda.to_cpu(x)
        for a in x:
            a_flat = a.flatten()
            a_new = self.transformer.fit_transform(a_flat.reshape(1, -1))
            embeddings.append(a_new)
        return self.xp.asarray(embeddings)
