# import util
from typing import overload


class layer(object):

    def __init__(self, layer_type):
        self.layer_type = layer_type

    def initialize(self, folder, name):
        pass

    def train(self, in_value):
        pass

    def eval(self, in_value):
        pass

    def update(self):
        pass

    def save_parameters(self):
        pass

    def load_parameters(self):
        pass

    def distributed_save_parameters(self):
        pass

    def distributed_load_parameters(self, param):
        pass

    def distributed_add_gradient(self, grad):
        pass

    def distributed_average_gradient(self, num):
        pass

    def distributed_save_gradient(self):
        pass

    def distributed_load_gradient(self,grad):
        pass
