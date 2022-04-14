from MiniFramework import *


class simple_cnn(NeuralNet):
    def __init__(self, params, model_name):
        super(simple_cnn, self).__init__(params,model_name)
        self.add_layer(ConLayer(1, 8, kernel_size=3, hp=params, stride=1), "c1")
        self.add_layer(ReLU(), "relu1")

        self.add_layer(PoolingLayer(kernel_size=2, stride=2, pooling_type=PoolingTypes.MAX), "p1")
        self.add_layer(ConLayer(8, 16, kernel_size=3, hp=params, stride=1), "c2")
        self.add_layer(ReLU(), "relu2")
        self.add_layer(PoolingLayer(kernel_size=2, stride=2, pooling_type=PoolingTypes.MAX), "p2")
        self.add_layer(FCLayer(400, 32, params), "f3")
        self.add_layer(BatchNormalLayer(32), "bn3")
        self.add_layer(ReLU(), "relu3")
        self.add_layer(FCLayer(32, 10, params), "f2")
        self.add_layer(Softmax(), "s4")

    def forward(self, input_v, train=True):
        output = None
        for i in range(self.layer_count):
            output = self.layer_list[i].forward(input_v, train)
            input_v = output

        self.output_v = output
        return self.output_v

    def backward(self, Y):
        delta_in = self.output_v - Y
        for i in range(self.layer_count - 1, -1, -1):
            layer = self.layer_list[i]
            delta_out = layer.backward(delta_in, i)
            delta_in = delta_out

    def update(self):
        for i in range(self.layer_count - 1, -1, -1):
            layer = self.layer_list[i]
            layer.update()

