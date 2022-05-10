from MiniFramework.Layer import *


class Add(layer):
    def __init__(self, x1, x2, layer_type):
        super().__init__(layer_type)
        self.x1 = x1
        self.x2 = x2

    def initialize(self, folder, name):
        pass
