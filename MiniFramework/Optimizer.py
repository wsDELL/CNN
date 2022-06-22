import numpy as np

from MiniFramework.util import *
from MiniFramework.Enums import *


class Optimizer(object):
    def __init__(self):
        pass

    def pre_update(self, theta):
        pass

    def update(self, theta, grad):
        pass


class SGD(Optimizer):
    def __init__(self, lr):
        super().__init__()
        self.lr = lr

    def update(self, theta, grad):
        theta = theta - self.lr * grad
        return theta


class Momentum_SGD(Optimizer):
    def __init__(self, lr):
        """
        :param lr: learning rate
        :param momentum
        :param vt(update value)
        """
        super().__init__()
        self.momentum = 0.9
        self.vt = 0.0
        self.lr = lr

    def update(self, theta, grad):
        vt_new = self.momentum * self.vt - self.lr * grad
        theta = theta + vt_new
        self.vt = vt_new
        return theta


class AdaGrad(Optimizer):
    def __init__(self, lr):
        """
        :param lr: learning rate
        :param eps:(optional) term added to the denominator to improve numerical stability (default: 1e-10)
        """
        super().__init__()
        self.lr = lr
        self.eps = 1e-10
        self.r = 0

    def update(self, theta, grad):
        self.r = self.r + np.multiply(grad, grad)
        theta = theta - self.lr * grad / (self.eps + np.sqrt(self.r))
        return theta


class RMSProp(Optimizer):
    def __init__(self, lr):
        super().__init__()
        self.lr = lr
        self.beta = 0.9
        self.eps = 1e-10
        self.r = 0

    def update(self, theta, grad):
        self.r = self.beta * self.r + (1 - self.beta) * grad * grad
        theta = theta - self.lr * grad / np.sqrt(self.eps + self.r)
        return theta


class AdaDelta(Optimizer):
    def __init__(self, lr):
        super().__init__()
        self.eps = 1e-5
        self.r = 0
        self.s = 0
        self.alpha = 0.9

    def update(self, theta, grad):
        grad2 = np.multiply(grad, grad)
        self.s = self.alpha * self.s + (1 - self.alpha) * grad2
        d_theta = np.sqrt((self.eps + self.r) / (self.eps + self.s)) * grad
        theta = theta - d_theta
        d_theta2 = np.multiply(d_theta, d_theta)
        self.r = self.alpha * self.r + (1 - self.alpha) * d_theta2
        return theta


class Adam(Optimizer):
    def __init__(self, lr=0.001):
        super().__init__()
        self.lr = lr
        self.p1 = 0.9
        self.p2 = 0.999
        self.eps = 1e-8
        self.t = 0.0
        self.m = 0.0
        self.v = 0.0

    def update(self, theta, grad):
        self.t = self.t + 1.0
        self.m = self.p1 * self.m + (1.0 - self.p1) * grad
        i = np.multiply(grad, grad)
        self.v = self.p2 * self.v + (1.0 - self.p2) * i
        m_hat = self.m / (1.0 - self.p1 ** self.t)
        v_hat = self.v / (1.0 - self.p2 ** self.t)
        d_theta = self.lr * m_hat / (self.eps + np.sqrt(v_hat))
        theta = theta - d_theta
        return theta


class OptimizerSelector(object):
    @staticmethod
    def CreateOptimizer(lr, name=OptimizerName.SGD):
        optimizer = None
        if name == OptimizerName.SGD:
            optimizer = SGD(lr)
        elif name == OptimizerName.Adam:
            optimizer = Adam(lr)
        elif name == OptimizerName.AdaGrad:
            optimizer = AdaGrad(lr)
        elif name == OptimizerName.Momentum:
            optimizer = Momentum_SGD(lr)
        elif name == OptimizerName.Nag:
            optimizer = RMSProp(lr)
        elif name == OptimizerName.RMSProp:
            optimizer = AdaDelta(lr)

        return optimizer
