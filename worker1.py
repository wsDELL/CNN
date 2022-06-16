import numpy as np


class Tensor(np.ndarray):
    def __init__(self, shape, dtype=None, buffer=None, offset=0, strides=None, order=None, required_grad=True):
        super(Tensor, self).__init__(shape)
        self.required_grad = required_grad
        self.grad_function = []
