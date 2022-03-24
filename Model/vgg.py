from MiniFramework import *
import numpy as np

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(NeuralNet):
    def __init__(self, param, vgg_name):
        super().__init__(param, vgg_name)
        self._make_layers(cfg[vgg_name])
        self.add_layer(DropoutLayer(), name="Drop1")
        self.add_layer(FCLayer(512, 512, param), name="fc1")
        self.add_layer(ReLU(), name="Relu_1")
        self.add_layer(DropoutLayer(), name="Drop2")
        self.add_layer(FCLayer(512, 512, param), name="fc2")
        self.add_layer(ReLU(), name="Relu_2")
        self.add_layer(FCLayer(512, 10, param), name="fc3")
        self.add_layer(Softmax(), name="softmax")

    def _make_layers(self, cfg: list):
        in_channels = 3
        M_count = 1
        con_count = 1
        for x in cfg:
            if x == "M":
                self.add_layer(PoolingLayer(kernel_size=2, stride=2), name=f"M{M_count}")
                M_count = M_count + 1
            else:
                self.add_layer(ConLayer(in_channels, x, kernel_size=3, hp=self.hp, padding=1), name=f"con{con_count}")
                self.add_layer(BatchNormalLayer(x), name=f"bn{con_count}")
                self.add_layer(ReLU(), name=f"relu{con_count}")
                con_count = con_count + 1
                in_channels = x
        # self.add_layer(PoolingLayer(kernel_size=1, stride=1, pooling_type=PoolingType.MEAN), name=f"AvgPool")

    def forward(self, input_v, train=True):
        output = None
        for i in range(self.layer_count):
            output = self.layer_list[i].forward(input_v, train)
            input_v = output

        self.output_v = output
        return self.output_v

    def backward(self, X, Y):
        delta_in = self.output_v - Y
        for i in range(self.layer_count - 1, -1, -1):
            layer = self.layer_list[i]
            delta_out = layer.backward(delta_in, i)
            delta_in = delta_out

    # def __pre_update(self):
    #     for i in range(self.layer_count - 1, -1, -1):
    #         layer = self.layer_list[i]
    #         layer.pre_update()

    def update(self):
        for i in range(self.layer_count - 1, -1, -1):
            layer = self.layer_list[i]
            layer.update()


if __name__ == "__main__":
    max_epoch = 20
    batch_size = 128
    learning_rate = 0.1
    params = HyperParameters(learning_rate, max_epoch, batch_size,
                             net_type=NetType.MultipleClassifier,
                             init_method=InitialMethod.Xavier,
                             optimizer_name=OptimizerName.SGD)
    net = VGG(param=params, vgg_name="VGG11")
    x = np.random.rand(2, 3, 32, 32)
    y = net.forward(x)
    z = net.backward()
    print(y.size())
