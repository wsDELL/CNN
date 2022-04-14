import math
import numpy as np


class Node(np.ndarray):
    _id = -1

    def __init__(self, op, inputs, shape):
        super().__init__(shape)
        self.inputs = inputs
        self.op = op
        self.grad = 0.0
        self.evaluate()
        self.id = Node._id
        Node._id += 1

    def evaluate(self):
        self.value = self.op.compute(self.input2value())

    def input2value(self):
        new_inputs = []
        for i in self.inputs:
            if isinstance(i, Node):
                i = i.value
            new_inputs.append(i)

        return new_inputs

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"Node{self.id}: {self.input2value()} {self.op.iteration_count()} = {self.value}, grad: {self.grad}"
