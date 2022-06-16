import numpy as np

from MiniFramework.Enums import *
from MiniFramework.Layer import *
from MiniFramework.Optimizer import *
from MiniFramework.WeightBias import *
from MiniFramework.HyperParameter import *


class FCLayer(layer):
    def __init__(self, input_n, output_n, hp, init_method=InitialMethod.Kaiming_Normal,layer_type="Fully connected "
                                                                                                  "layer"):
        super().__init__(layer_type)
        self.input_num = input_n
        self.output_num = output_n
        self.WB = WeightsBias(self.input_num, self.output_num,  hp.optimizer_name, hp.lr,init_method=init_method)
        self.regular_name = hp.regular_name
        self.regular_value = hp.regular_value
        self.input_v = None
        self.output_v = None
        self.input_shape = None
        self.name = None

    def initialize(self, folder, name):
        self.WB.initialize(folder, name, True)
        self.name = name

    def forward(self, input_v: np.ndarray, train=True):

        self.input_shape = input_v.shape
        if input_v.ndim == 4:
            self.input_v = input_v.reshape(self.input_shape[0], -1)
        else:
            self.input_v = input_v
        self.output_v = np.dot(self.input_v, self.WB.W) + self.WB.B
        return self.output_v

    def backward(self, delta_in, layer_idx):
        dZ = delta_in
        m = self.input_v.shape[0]
        if self.regular_name == RegularMethod.L2:
            self.WB.dW = (np.dot(self.input_v.T, dZ) + self.regular_value * self.WB.W) / m
        elif self.regular_name == RegularMethod.L1:
            self.WB.dW = (np.dot(self.input_v.T, dZ) + self.regular_value * np.sign(self.WB.W)) / m
        else:
            self.WB.dW = np.dot(self.input_v.T, dZ) / m
        self.WB.dB = np.sum(dZ, axis=0, keepdims=True) / m
        if layer_idx == 0:
            return None
        delta_out = np.dot(dZ, self.WB.W.T)

        if len(self.input_shape) > 2:
            return delta_out.reshape(self.input_shape)
        else:
            return delta_out

    def update(self):
        self.WB.Update()

    def save_parameters(self):
        self.WB.SaveResultValue()

    def load_parameters(self):
        self.WB.LoadResultValue()

    def distributed_save_parameters(self):
        param = self.WB.distributed_SaveParameter()
        return param

    def distributed_load_parameters(self, param: dict):
        self.WB.distributed_LoadParameter(param)

    def distributed_save_gradient(self):
        grad = self.WB.distributed_SaveGradient()
        return grad

    def distributed_load_gradient(self,grad):
        self.WB.distributed_LoadGradient(grad)

    def distributed_add_gradient(self, grad: dict):
        self.WB.distributed_AddGradient(grad)

    def distributed_average_gradient(self, num):
        self.WB.distributed_AverageGradient(num)




