import math
from MiniFramework.Enums import *
from MiniFramework.WeightBias import *
from MiniFramework.Optimizer import *


class ConWeightBias(WeightsBias):
    def __init__(self, input_c, output_c, filter_w, filter_h, init_method, optimizer_name, eta):
        self.FilterCount = output_c
        self.KernalCount = input_c
        self.KernalHeight = filter_h
        self.KernalWidth = filter_w
        self.init_method = init_method
        self.optimizer_name = optimizer_name
        self.lr = eta
        self.name = None
        self.WBShape = (self.FilterCount, self.KernalCount, self.KernalHeight, self.KernalWidth)


    def initialize(self, folder, name, create_new):
        self.init_file_name = f"{folder}/{name}_{self.FilterCount}_" \
                              f"{self.KernalCount}_{self.KernalHeight}_{self.KernalWidth}_init.npz"
        self.result_file_name = f"{folder}/{name}_result.npz"
        self.name = name
        if create_new:
            self.CreateNew()
        else:
            self.LoadExistingParameter()

        self.CreateOptimizers()
        self.dW = np.zeros(self.W.shape).astype('float32')
        self.dB = np.zeros(self.B.shape).astype('float32')

    def CreateNew(self):
        self.W = ConWeightBias.InitialConvParameters(self.WBShape,self.init_method)
        self.B = np.zeros((self.FilterCount, 1)).astype('float32')


    def Rotate180(self):
        self.WT = np.zeros(self.W.shape).astype(np.float32)
        for i in range(self.FilterCount):
            for j in range(self.KernalCount):
                self.WT[i, j] = np.rot90(self.W[i, j], 2)
        return self.WT

    def ClearGrads(self):
        self.dW = np.zeros(self.W.shape).astype(np.float32)
        self.dB = np.zeros(self.B.shape).astype(np.float32)

    def MeanGrads(self, m):
        self.dW = self.dW / m
        self.dB = self.dB / m


    @staticmethod
    def InitialConvParameters(shape, init_method):
        assert (len(shape) == 4)
        num_input = shape[2]
        num_output = shape[3]

        if init_method == InitialMethod.Zero:
            W = np.zeros(shape).astype('float32')
        elif init_method == InitialMethod.Normal:
            W = np.random.normal(shape).astype('float32')
        elif init_method == InitialMethod.MSRA:
            W = np.random.normal(0, np.sqrt(2 / num_input * num_output), shape).astype('float32')
        elif init_method == InitialMethod.Xavier:
            t = math.sqrt(6 / (num_output + num_input))
            W = np.random.uniform(-t, t, shape).astype('float32')
        return W




