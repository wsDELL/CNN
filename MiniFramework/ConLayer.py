from MiniFramework.ConWeightBias import *
from MiniFramework.Layer import *
from MiniFramework.MnistDataReader import *


# import numpy as np


class ConLayer(layer):
    def __init__(self, in_planes, out_planes, kernel_size: int, hp, layer_type='', stride=1, padding=0):
        super().__init__(layer_type)
        self.input_channel = in_planes
        # self.input_width = input_shape[1]
        # self.input_height = input_shape[2]
        self.output_channel = out_planes
        self.filter_width = kernel_size
        self.filter_height = kernel_size
        self.stride = stride
        self.padding = padding
        self.hp = hp
        self.input_v = None
        self.batch_size = None
        self.output_v = None
        self.name = None

    def initialize(self, folder, name, create_new=True):
        self.WB = ConWeightBias(self.input_channel, self.output_channel, self.filter_height, self.filter_width,
                                self.hp.init_method, self.hp.optimizer_name, self.hp.lr)
        self.WB.initialize(folder, name, create_new)
        self.name = name

    def forward(self, input_v, train=True):
        self.input_v = input_v
        return self._forward_img2col(input_v)

    def backward(self, delta_in, layer_idx):
        delta_out, dw, db = self._backward_col2img(delta_in, layer_idx)
        return delta_out

    def _forward_img2col(self, input_v: np.ndarray):
        assert (self.input_v.ndim == 4)
        self.input_width = input_v.shape[2]
        self.input_height = input_v.shape[3]

        _output_shape = calculate_output_size(self.input_height, self.input_width, self.filter_height,
                                              self.filter_width,
                                              self.padding, stride=self.stride)
        self.output_shape = (self.output_channel, _output_shape[0], _output_shape[1])
        self.batch_size = self.input_v.shape[0]
        # assert (self.input_v.shape == (self.batch_size, self.input_channel, self.input_height, self.input_width))
        self.col_x = img2col(input_v, self.filter_height, self.filter_width, self.stride, self.padding)
        self.col_w = self.WB.W.reshape(self.output_channel, -1).T
        self.col_b = self.WB.B.reshape(-1, self.output_channel)
        out1 = np.dot(self.col_x, self.col_w) + self.col_b
        out2 = out1.reshape(self.batch_size, self.output_shape[1], self.output_shape[2], -1)
        self.output_v = np.transpose(out2, axes=(0, 3, 1, 2))
        return self.output_v

    def _backward_col2img(self, delta_in, layer_idx):
        col_delta_in = np.transpose(delta_in, axes=(0, 2, 3, 1)).reshape(-1, self.output_channel)
        self.WB.dB = np.sum(col_delta_in, axis=0, keepdims=True).T / self.batch_size
        col_dW = np.dot(self.col_x.T, col_delta_in) / self.batch_size
        self.WB.dW = np.transpose(col_dW, axes=(1, 0)).reshape(self.output_channel, self.input_channel,
                                                               self.filter_height, self.filter_width)
        col_delta_out = np.dot(col_delta_in, self.col_w.T)
        delta_out = col2img(col_delta_out, self.input_v.shape, self.filter_height, self.filter_width, self.stride,
                            self.padding, self.output_shape[1],
                            self.output_shape[2])
        return delta_out, self.WB.dW, self.WB.dB

    def pre_update(self):
        pass

    def update(self):
        self.WB.Update()

    def save_parameters(self):
        self.WB.SaveResultValue()

    def load_parameters(self):
        self.WB.LoadResultValue()

    def distributed_save_parameters(self):
        param = self.WB.distributed_SaveResultValue()
        return param

    def distributed_load_parameters(self, param):
        self.WB.distributed_LoadResultValue(param)

    def distributed_add_parameters(self, param):
        self.WB.distributed_AddResultValue(param)

    def distributed_average_parameters(self, num):
        self.WB.distributed_AverageResultValue(num)
