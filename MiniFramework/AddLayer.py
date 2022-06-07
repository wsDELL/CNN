from MiniFramework.Layer import *


class Add(layer):
    def __init__(self, layer_type="add layer"):
        super().__init__(layer_type)
        self.result_file_name = ''
        self.name = None

    def initialize(self, folder, name):
        self.result_file_name = f'{folder}/{name}_result.npz'
        self.name = name

    @staticmethod
    def forward(input_v, residual, train=True):
        result = input_v + residual
        return result

    @staticmethod
    def backward(delta_in):
        res = [delta_in, delta_in]
        return res
