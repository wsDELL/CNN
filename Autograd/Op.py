from Autograd.Node import *


class Op(object):
    def name(self):
        pass

    def __call__(self, *args, **kwargs):
        pass

    def compute(self, inputs):
        pass

    def gradient(self, inputs, output_grad):
        pass


class AddOp(Op):
    def name(self):
        return "add"

    def __call__(self, a, b):
        return Node(self, [a, b])

    def compute(self, inputs):
        return inputs[0] + inputs[1]

    def gradient(self, inputs, output_grad):
        return [output_grad, output_grad]


class SupOp(Op):
    def name(self):
        return
