from MiniFramework import *


class dis_AlexNet(NeuralNet):
    def __init__(self, param, model_name):
        super().__init__(param, model_name)
        self.add_layer(ConLayer(3, 64, kernel_size=3, hp=param, stride=2, padding=1), "c1")
        self.add_layer(ReLU(), "relu1")
        self.add_layer(PoolingLayer(kernel_size=2, stride=2, pooling_type=PoolingTypes.MAX), "p1")
        self.add_layer(ConLayer(64, 192, kernel_size=3, hp=param, stride=1, padding=1), "c2")
        self.add_layer(ReLU(), "relu2")
        self.add_layer(PoolingLayer(kernel_size=2, stride=2, pooling_type=PoolingTypes.MAX), "p2")
        self.add_layer(ConLayer(192, 384, kernel_size=3, hp=param, stride=1, padding=1), "c3")
        self.add_layer(ReLU(), "relu3")
        self.add_layer(ConLayer(384, 256, kernel_size=3, hp=param, stride=1, padding=1), "c4")
        self.add_layer(ReLU(), "relu4")
        self.add_layer(ConLayer(256, 256, kernel_size=3, hp=param, stride=1, padding=1), "c5")
        self.add_layer(ReLU(), "relu5")
        self.add_layer(PoolingLayer(kernel_size=2, stride=2, pooling_type=PoolingTypes.MEAN), "p5")
        self.add_layer(DropoutLayer(ratio=0.3), 'd1')
        self.add_layer(FCLayer(256 * 2 * 2, 1024, param), "f1")
        self.add_layer(ReLU(), "relu6")
        self.add_layer(DropoutLayer(ratio=0.3), "d2")
        self.add_layer(FCLayer(1024, 1024, param), "f2")
        self.add_layer(ReLU(), "relu7")
        self.add_layer(FCLayer(1024, 10, param), "f3")
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


