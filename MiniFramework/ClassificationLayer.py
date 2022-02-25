import numpy as np

from MiniFramework.Layer import *
from MiniFramework.util import *


class ClassificationLayer(layer):
    def __init__(self, layer_type="classification layer"):
        super().__init__(layer_type)
        self.input_v = None
        self.a = None

    def forward(self, input_v, train=True):
        self.input_v = input_v
        self.a = self._forward(self.input_v)
        return self.a

    def backward(self, delta_in, flag):
        dZ = delta_in
        return dZ

    def _forward(self, input_v):
        pass


class Classifier(object):
    def forward(self, z):
        pass


class Softmax(ClassificationLayer):
    def _forward(self, z):
        shift_z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(shift_z)
        a = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        return a


class Logistic(ClassificationLayer):
    def _forward(self, z):
        a = 1.0 / (1.0 + np.exp(-z))
        return a
