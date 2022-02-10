import numpy as np

from MiniFramework.Layer import *
from MiniFramework.util import *


class ActivationLayer(layer):

    def __init__(self, activator, layer_type="activate function"):
        super().__init__(layer_type)
        self.input_v = None
        self.a = None
        self.activator = activator
        self.output_shape = None

    def forward(self, input_v, train=True):
        self.input_v = input_v
        self.a = self.activator.forward(self.input_v)
        self.output_shape = self.a.shape
        return self.a

    def backward(self, delta_in, layer_idx):
        dZ = self.activator.backward(self.input_v, self.activator, delta_in)
        return dZ


class Activation_function(object):
    def __init__(self):
        pass

    def forward(self, z):
        pass

    def backward(self, z, a, delta):
        pass


class Identity(Activation_function):
    def forward(self, z):
        return z

    def backward(self, z, a, delta):
        return delta


class Sigmoid(Activation_function):
    def forward(self, z):
        a = 1.0 / (1.0 + np.exp(-z))
        return a

    def backward(self, z, a, delta):
        da = np.multiply(a, 1 - a)
        dz = np.multiply(delta, da)
        return dz


class Tanh(Activation_function):
    def forward(self, z):
        a = (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))
        return a

    def backward(self, z, a, delta):
        da = 1 - np.multiply(a, a)
        dz = np.multiply(delta, da)
        return dz


class ReLU(Activation_function):
    def forward(self, z):
        a = np.maximum(z, 0)
        return a

    def backward(self, z, a, delta):
        da = np.zeros(z.shape).astype('float32')
        da[z > 0] = 1
        dz = da * delta
        return dz


class LeakyReLU(Activation_function):
    def forward(self, z):
        a = np.maximum(0.01*z, z)
        return a

    def backward(self, z, a, delta):
        da = np.zeros(z.shape).astype('float32')
        da[z > 0.01] = 1
        dz = da * delta
        return dz
