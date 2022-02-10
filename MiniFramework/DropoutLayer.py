from MiniFramework.Layer import *
from MiniFramework.util import *


class DropoutLayer(layer):
    def __init__(self, input_size, ratio=0.5):
        self.dropout_ratio = ratio
        self.input_size = input_size
        self.output_size = input_size
        self.mask = None
        self.output_v = None

    def forward(self, input_v: np.ndarray, train=True):
        assert (input_v.ndim == 2)
        if train:
            self.mask = np.random.rand(*input_v.shape) > self.dropout_ratio
            self.output_v = input_v * self.mask
        else:
            self.output_v = input_v * (1.0 - self.dropout_ratio)

        return self.output_v

    def backward(self, delta_in, idx):
        delta_out = self.mask * delta_in
        return delta_out



