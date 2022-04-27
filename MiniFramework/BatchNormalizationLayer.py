import numpy
import numpy as np

from MiniFramework.Layer import *
from MiniFramework.util import *


class BatchNormalLayer(layer):
    def __init__(self, input_size, layer_type='Batch Normalization', momentum=0.9):
        super().__init__(layer_type)

        self.gamma = np.ones((1, input_size)).astype('float32')
        self.beta = np.zeros((1, input_size)).astype('float32')
        self.eps = 1e-5
        self.input_size = input_size
        self.output_size = input_size
        self.momentum = momentum
        self.running_mean = np.zeros((1, input_size)).astype('float32')
        self.running_variance = np.zeros((1, input_size)).astype('float32')
        self.input_v = None
        self.output_v = None
        self.mean = None
        self.variance = None
        self.input_normed = None
        self.std = None
        self.d_beta = None
        self.d_gamma = None
        self.result_file_name = ''
        self.name = None
        self.counter = True

    def initialize(self, folder, name, create_new=False):
        self.result_file_name = f'{folder}/{name}_result.npz'
        self.name = name

    def forward(self, input_v, train=True):
        assert (input_v.ndim == 2 or input_v.ndim == 4)# fc or cv
        if input_v.ndim == 4 and self.counter is True:
            self.input_width = input_v.shape[2]
            self.input_height = input_v.shape[3]
            self.gamma = np.ones((1, self.input_size, self.input_width,self.input_height)).astype('float32')
            self.beta = np.zeros((1, self.input_size, self.input_width,self.input_height)).astype('float32')
            self.running_mean = np.zeros((1, self.input_size, self.input_width,self.input_height)).astype('float32')
            self.running_variance = np.zeros((1, self.input_size, self.input_width,self.input_height)).astype('float32')
            self.counter = False
        self.input_v = input_v

        if train:
            self.mu = np.mean(self.input_v, axis=0, keepdims=True)
            self.x_mu = self.input_v - self.mu
            self.variance = np.mean(self.x_mu ** 2, axis=0, keepdims=True) + self.eps
            self.std = np.sqrt(self.variance)
            self.norm_x = self.x_mu / self.std
            self.z = self.gamma * self.norm_x + self.beta
            self.running_mean = self.momentum * self.running_mean + (1.0 - self.momentum) * self.mu
            self.running_variance = self.momentum * self.running_variance + (1.0 - self.momentum) * self.variance
        else:
            self.mu = self.running_mean
            self.variance = self.running_variance
            self.norm_x = (self.input_v - self.mu) / np.sqrt(self.variance + self.eps)
            self.z = self.gamma * self.norm_x + self.beta
        # end if
        return self.z

    def backward(self, delta_in: np.ndarray, flag):
        assert (delta_in.ndim == 2 or delta_in.ndim == 4)
        m = self.input_v.shape[0]
        self.d_gamma = np.sum(delta_in * self.norm_x, axis=0, keepdims=True)
        self.d_beta = np.sum(delta_in, axis=0, keepdims=True)
        d_input_normed = self.gamma * delta_in
        d_variance = -0.5 * np.sum(d_input_normed * self.x_mu, axis=0, keepdims=True) / (
                self.variance * self.std)
        d_mean = -np.sum(d_input_normed / self.std, axis=0, keepdims=True) - 2 / m * d_variance * np.sum(self.x_mu,
                                                                                                         axis=0,
                                                                                                         keepdims=True)
        delta_out = d_input_normed / self.std + d_variance * 2 * self.x_mu / m + d_mean / m

        if flag == -1:
            return delta_out, self.d_gamma, self.d_beta
        else:
            return delta_out

    def update(self, learning_rate=0.1):
        self.gamma = self.gamma - learning_rate * self.d_gamma
        self.beta = self.beta - learning_rate * self.d_beta

    def save_parameters(self):
        np.savez(self.result_file_name, gamma=self.gamma, beta=self.beta,
                 mean=self.running_mean, variance=self.running_variance)

    def load_parameters(self):
        data = np.load(self.result_file_name)
        self.gamma = data['gamma']
        self.beta = data['beta']
        self.running_mean = data['mean']
        self.running_variance = data['variance']

    def distributed_save_parameters(self):
        params = {self.name: {"gamma": self.gamma, "beta": self.beta}}
        return params

    def distributed_load_parameters(self, param):
        # iteration_count = list(param.keys())
        self.gamma = param[self.name]['gamma']
        self.beta = param[self.name]['beta']
        # self.running_mean = param[self.name]['mean']
        # self.running_variance = param[self.name]['variance']

    def distributed_save_gradient(self):
        grad = {self.name: {"d_gamma": self.d_gamma, "d_beta": self.d_beta}}
        return grad

    def distributed_load_gradient(self, grad):
        self.d_gamma = grad[self.name]['d_gamma']
        self.d_beta = grad[self.name]['d_beta']

    def distributed_add_gradient(self, grad):
        # iteration_count = list(param.keys())
        self.d_gamma = self.d_gamma + grad[self.name]['d_gamma']
        self.d_beta = self.beta + grad[self.name]['d_beta']
        # self.running_mean = self.running_mean + grad[self.name]['mean']
        # self.running_variance = self.running_variance + grad[self.name]['variance']

    def distributed_average_gradient(self, num):
        self.d_gamma = self.d_gamma/num
        self.d_beta = self.d_beta/num
        # self.running_mean = self.running_mean/num
        # self.running_variance = self.running_variance/num
