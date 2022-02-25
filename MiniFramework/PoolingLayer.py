import numpy as np

from MiniFramework.util import *
from MiniFramework.Layer import *
from MiniFramework.Enums import *


class PoolingLayer(layer):
    def __init__(self, kernel_size: int, stride=1, padding=0, pooling_type=PoolingType.MAX, layer_type='Pooling layer'):
        super().__init__(layer_type)

        self.stride = stride
        self.padding = padding
        self.pooling_type = pooling_type
        self.pool_height = kernel_size
        self.pool_width = kernel_size
        self.pool_size = self.pool_width * self.pool_height
        self.init_file_name = None
        self.input_v = None
        self.arg_max = None
        self.output_v = None
        self.name = None

    def initialize(self, folder, name):
        self.init_file_name = f"{folder}/{name}_init.npz"
        self.name = name

    def forward(self, input_v, train=True):
        return self.forward_img2col(input_v, train)
        # return self.forward_numba(input_v, train)

    def backward(self, delta_in, layer_idx):
        return self.backward_col2img(delta_in, layer_idx)
        # return self.backward_numba(delta_in, layer_idx)

    def forward_img2col(self, input_v, train=True):
        self.input_v = input_v
        N, C, H, W = input_v.shape
        self.num_input_channel = C
        self.input_height = H
        self.input_width = W
        self.output_width = (self.input_width - self.pool_width) // self.stride + 1
        self.output_height = (self.input_height - self.pool_height) // self.stride + 1
        self.output_shape = (self.num_input_channel, self.output_height, self.output_width)
        self.output_size = self.num_input_channel * self.output_height * self.output_width
        col = img2col(input_v, self.pool_height, self.pool_width, self.stride, self.padding)
        col_x = col.reshape(-1, self.pool_height * self.pool_width)
        self.arg_max = np.argmax(col_x, axis=1)
        if self.pooling_type == PoolingType.MAX:
            out1 = np.max(col_x, axis=1)
        elif self.pooling_type == PoolingType.MEAN:
            out1 = np.mean(col_x, axis=1)
        else:
            out1 = np.max(col_x, axis=1)
        out2 = out1.reshape(N, self.output_height, self.output_width, C)
        self.output_v = np.transpose(out2, axes=(0, 3, 1, 2))
        return self.output_v

    def backward_col2img(self, delta_in: np.ndarray, layer_idx):
        dout = np.transpose(delta_in, (0, 2, 3, 1))
        dmax = np.zeros((dout.size, self.pool_size)).astype('float32')
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (self.pool_size,))
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2img(dcol, self.input_v.shape, self.pool_height, self.pool_width, self.stride, 0, self.output_height,
                     self.output_width)
        return dx

    def forward_numba(self, input_v, train=True):
        assert (input_v.ndim == 4)
        self.input_v = input_v
        N, C, H, W = input_v.shape
        self.num_input_channel = C
        self.input_height = H
        self.input_width = W
        self.output_width = (self.input_width - self.pool_width) // self.stride + 1
        self.output_height = (self.input_height - self.pool_height) // self.stride + 1
        self.output_shape = (self.num_input_channel, self.output_height, self.output_width)
        self.output_size = self.num_input_channel * self.output_height * self.output_width
        self.batch_size = self.input_v.shape[0]
        self.z = jit_maxpool_forward(self.input_v, self.batch_size, self.num_input_channel, self.output_height,
                                     self.output_width, self.pool_height, self.pool_width, self.stride)
        return self.z

    def backward_numba(self, delta_in, layer_idx):
        assert (delta_in.ndim == 4)
        assert (delta_in.shape == self.z.shape)
        delta_out = jit_maxpool_backward(self.input_v, delta_in, self.batch_size, self.num_input_channel,
                                         self.output_height,
                                         self.output_width, self.pool_height, self.pool_width, self.stride)
        return delta_out

    def save_parameters(self):
        np.savez(self.init_file_name, self.pooling_type)

    def load_parameters(self):
        self.mode = np.load(self.init_file_name, allow_pickle=True)
        pass

    def distributed_save_parameters(self):
        dis = {self.name: {'pooling_type': self.pooling_type}}
        return dis

    def distributed_load_parameters(self, param: dict):
        self.pooling_type = param[self.name]['pooling_type']

    def distributed_add_parameters(self, param: dict):
        pass

    def distributed_average_parameters(self, num):
        pass
