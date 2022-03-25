import math
from MiniFramework.Enums import *
from MiniFramework.WeightBias import *
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
    def InitialConvParameters(shape, init_method, nonlinearity='conv2d'):
        assert (len(shape) == 4)
        num_input = shape[1]
        num_output = shape[0]

        if init_method == InitialMethod.Zero:
            W = np.zeros(shape).astype('float32')
        elif init_method == InitialMethod.Uniform:
            W = np.random.uniform(shape).astype('float32')
        elif init_method == InitialMethod.Normal:
            W = np.random.normal(shape).astype('float32')
        elif init_method == InitialMethod.MSRA:
            W = np.random.normal(0, np.sqrt(2 / num_input * num_output), shape).astype('float32')
        elif init_method == InitialMethod.Xavier_Uniform:
            gain = calculate_gain(nonlinearity)
            t = gain * math.sqrt(6.0 / float(num_output + num_input))
            W = np.random.uniform(-t, t, shape).astype('float32')
        elif init_method == InitialMethod.Xavier_Normal:
            gain = calculate_gain(nonlinearity)
            t = gain * math.sqrt(2.0 / float(num_output + num_input))
            W = np.random.normal(0., t, shape).astype('float32')
        elif init_method == InitialMethod.Kaiming_Uniform:
            gain = calculate_gain(nonlinearity)
            std = gain / math.sqrt(num_input)
            bound = math.sqrt(3.0) * std
            W = np.random.uniform(-bound, bound, shape).astype('float32')
        elif init_method == InitialMethod.Kaiming_Normal:
            gain = calculate_gain(nonlinearity)
            std = gain / math.sqrt(num_input)
            W = np.random.normal(0, std, shape).astype('float32')
        return W




