import math

import numpy as np

from MiniFramework.util import *
from MiniFramework.Enums import *
from MiniFramework.Optimizer import *


class WeightsBias(object):

    def __init__(self, n_input, n_output, init_method, optimizer_name, eta):
        self.result_file_name = None
        self.init_file_name = None
        self.num_input = n_input
        self.num_output = n_output
        self.init_method = init_method
        self.optimizer_name = optimizer_name
        self.lr = eta
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
        self.W = WeightsBias.InitialParameters(self.num_input, self.num_output, self.init_method)
        self.B = np.zeros((1, self.num_output)).astype('float32')
        self.dW = np.zeros(self.W.shape).astype('float32')
        self.dB = np.zeros(self.B.shape).astype('float32')

    @staticmethod
    def InitialParameters(num_input, num_output, init_method):
        if init_method == InitialMethod.Zero:
            W = np.zeros((num_input, num_output)).astype('float32')
        elif init_method == InitialMethod.Normal:
            W = np.random.normal(size=(num_input, num_output)).astype('float32')
        elif init_method == InitialMethod.MSRA:
            W = np.random.normal(0, np.sqrt(2 / num_output), size=(num_input, num_output)).astype('float32')
        elif init_method == InitialMethod.Xavier:
            t = math.sqrt(6 / (num_output + num_input))
            W = np.random.uniform(-t, t, (num_input, num_output)).astype('float32')

        return W

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

    def distributed_SaveResultValue(self):
        dis = {self.name: {"W": self.W, "B": self.B}}
        return dis

    def distributed_LoadResultValue(self, param: dict):
        # _param = list(param.keys())
        self.W = param[self.name]['W']
        self.B = param[self.name]['B']

    def distributed_AddResultValue(self, param: dict):
        # _param = list(param.keys())
        self.W = self.W + param[self.name]['W']
        self.B = self.B + param[self.name]['B']

    def distributed_AverageResultValue(self, num):
        self.W = self.W/num
        self.B = self.B/num


            
