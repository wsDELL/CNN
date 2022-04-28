import math
from pathlib import Path

import numpy as np

from MiniFramework.util import *
from MiniFramework.Enums import *
from MiniFramework.Optimizer import *


def calculate_gain(nonlinearity, param=None):
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d',
                  'conv_transpose3d']
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        return 1
    elif nonlinearity == 'tanh':
        return 5.0 / 3
    elif nonlinearity == 'relu':
        return math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            negative_slope = param
        else:
            raise ValueError("negative_slope {} not a valid number".format(param))
        return math.sqrt(2.0 / (1 + negative_slope ** 2))
    elif nonlinearity == 'selu':
        return 3.0 / 4
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))


class WeightsBias(object):

    def __init__(self, n_input, n_output, optimizer_name, lr,init_method=InitialMethod.Normal):
        self.result_file_name = None
        self.init_file_name = None
        self.num_input = n_input
        self.num_output = n_output
        self.init_method = init_method
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.W = None
        self.B = None
        self.oB = None
        self.oW = None
        self.dW = None
        self.dB = None
        self.name = None
        self.CreateOptimizers()

    def initialize(self, folder, name, create_new):
        self.init_file_name = f"{folder}/{name}_{self.num_input}_{self.num_output}.npz"
        self.result_file_name = f"{folder}/{name}_result.npz"
        self.name = name
        if create_new:
            self.CreateNew()
        else:
            self.LoadExistingParameter()

    def CreateNew(self):
        self.W = WeightsBias.InitialParameters(self.num_input, self.num_output, nonlinearity='linear',
                                               init_method=self.init_method)
        self.B = np.zeros((1, self.num_output)).astype('float32')
        self.dW = np.zeros(self.W.shape).astype('float32')
        self.dB = np.zeros(self.B.shape).astype('float32')

    @staticmethod
    def InitialParameters(num_input, num_output, init_method, nonlinearity='linear'):
        if init_method == InitialMethod.Zero:
            Weight = np.zeros((num_input, num_output)).astype('float32')
        elif init_method == InitialMethod.Uniform:
            Weight = np.random.uniform(size=(num_input, num_output)).astype('float32')
        elif init_method == InitialMethod.Normal:
            Weight = np.random.normal(size=(num_input, num_output)).astype('float32')
        elif init_method == InitialMethod.MSRA:
            Weight = np.random.normal(0, np.sqrt(2 / num_output), size=(num_input, num_output)).astype('float32')
        elif init_method == InitialMethod.Xavier_Uniform:
            gain = calculate_gain(nonlinearity)
            t = gain * math.sqrt(6.0 / float(num_output + num_input))
            Weight = np.random.uniform(-t, t, size=(num_input, num_output)).astype('float32')
        elif init_method == InitialMethod.Xavier_Normal:
            gain = calculate_gain(nonlinearity)
            t = gain * math.sqrt(2.0 / float(num_output + num_input))
            Weight = np.random.normal(0., t, size=(num_input, num_output)).astype('float32')
        elif init_method == InitialMethod.Kaiming_Uniform:
            gain = calculate_gain(nonlinearity)
            std = gain / math.sqrt(num_input)
            bound = math.sqrt(3.0) * std
            Weight = np.random.uniform(-bound, bound, size=(num_input, num_output)).astype('float32')
        elif init_method == InitialMethod.Kaiming_Normal:
            gain = calculate_gain(nonlinearity)
            std = gain / math.sqrt(num_input)
            Weight = np.random.normal(0, std, size=(num_input, num_output)).astype('float32')

        return Weight

    def CreateOptimizers(self):
        self.oW = OptimizerSelector.CreateOptimizer(self.lr, self.optimizer_name)
        self.oB = OptimizerSelector.CreateOptimizer(self.lr, self.optimizer_name)

    def Update(self):
        self.W = self.oW.update(self.W, self.dW)
        self.B = self.oB.update(self.B, self.dB)

    def ClearGrads(self):
        self.dW = np.zeros(self.W.shape).astype(np.float32)
        self.dB = np.zeros(self.B.shape).astype(np.float32)

    def MeanGrads(self, m):
        self.dW = self.dW / m
        self.dB = self.dB / m

    def LoadExistingParameter(self):
        input_file = Path(self.init_file_name)
        if input_file.exists():
            self.LoadInitialValue()
        else:
            self.CreateNew()

    def SaveInitialValue(self):
        np.savez(self.init_file_name, W=self.W, B=self.B)

    def LoadInitialValue(self):
        data = np.load(self.init_file_name)
        self.W = data['W']
        self.B = data['B']

    def SaveResultValue(self):
        np.savez(self.result_file_name, W=self.W, B=self.B)

    def LoadResultValue(self):
        data = np.load(self.result_file_name)
        self.W = data["W"]
        self.B = data["B"]

    def distributed_SaveGradient(self):
        grad = {self.name: {"dW": self.dW, "dB": self.dB}}
        return grad

    def distributed_SaveParameter(self):
        dis = {self.name: {"W": self.W, "B": self.B}}
        return dis

    def distributed_LoadGradient(self, grad: dict):
        # _param = list(param.keys())
        self.dW = grad[self.name]['dW']
        self.dB = grad[self.name]['dB']

    def distributed_LoadParameter(self, param: dict):
        # _param = list(param.keys())
        self.W = param[self.name]['W']
        self.B = param[self.name]['B']

    def distributed_AddGradient(self, param: dict):
        # _param = list(param.keys())
        self.dW = self.dW + param[self.name]['dW']
        self.dB = self.dB + param[self.name]['dB']

    def distributed_AverageGradient(self, num):
        self.dW = self.dW / num
        self.dB = self.dB / num

    def _calculate_fan_in_and_fan_out(self, tensor: np.ndarray):
        dimensions = tensor.ndim
        if dimensions < 2:
            raise ValueError("Fan in and Fan out can not be computed for tensor with less than 2 dimensions")
        if dimensions == 2:
            fan_in = tensor.shape[1]
            fan_out = tensor.shape[0]
        else:
            num_input_fmaps = tensor.shape[1]
            num_output_fmaps = tensor.shape[0]
            receptive_field_size = 1
            if tensor.ndim > 2:
                receptive_field_size = tensor.size / (tensor.shape[0] * tensor.shape[1])
            fan_in = num_input_fmaps * receptive_field_size
            fan_out = num_output_fmaps * receptive_field_size

        return fan_in, fan_out
