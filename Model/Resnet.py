import MiniFramework
from MiniFramework import *


def conv3x3(in_planes: int, out_planes: int, param, stride: int = 1, padding: int = 1, ) -> MiniFramework.ConLayer:
    return MiniFramework.ConLayer(in_planes=in_planes, out_planes=out_planes, kernel_size=3, hp=param, stride=stride,
                                  padding=padding)


def con1x1(in_planes: int, out_planes: int, param, stride: int = 1, ) -> MiniFramework.ConLayer:
    return MiniFramework.ConLayer(in_planes=in_planes, out_planes=out_planes, kernel_size=1, hp=param, stride=stride,
                                  )


class BasicBlock(MiniFramework.NeuralNet):
    expansion = 1

    def __init__(self, in_planes, planes, stride, param, layer_name, downsample: dict = None):
        super().__init__(param, layer_name)

        self.add_layer(conv3x3(in_planes, planes, param=param, stride=stride), layer_name + "_con1")
        self.add_layer(MiniFramework.BatchNormalLayer(planes), layer_name + "_bn1")
        self.add_layer(MiniFramework.ActivationLayer(ReLU()), layer_name + "_relu1")
        self.add_layer(conv3x3(in_planes, planes, param=param), layer_name + "_con2")
        self.add_layer(BatchNormalLayer(planes), layer_name + "_bn2")
        self.downsample = downsample
        if self.downsample is not None:
            self.downsample_status = True
            for name in list(downsample.keys()):
                self.add_layers(downsample[name], layer_name + "_" + name)
        else:
            self.downsample_status = False
        self.add_layer(ActivationLayer(ReLU()), layer_name + "_relu2")
        self.stride = stride

    def forward(self, input_v, train=True):
        output = None
        residual = input_v
        layer_name: str = self.layer_list[0].name
        if self.downsample_status is True:
            for i in range(self.layer_count):
                if layer_name[-15:] == "_downsample_con":
                    residual = self.layer_list[i].forward(input_v, train)
                if layer_name[-14:] == "_downsample_bn":
                    residual = self.layer_list[i].forward(residual, train)
                    output = output + residual
                else:
                    output = self.layer_list[i].forward(input_v, train)
        else:
            for j in range(self.layer_count):
                output = self.layer_list[j].forward(input_v, train)
                if layer_name[-4:] == "_bn2":
                    output = output + residual
        self.output_v = output
        return self.output_v

    def backward(self, X, Y):
        delta_in = self.output_v - Y
        for i in range(self.layer_count - 1, -1, -1):
            layer = self.layer_list[i]
            delta_out = layer.backward(delta_in, i)
            delta_in = delta_out

    def update(self):
        for i in range(self.layer_count - 1, -1, -1):
            layer = self.layer_list[i]
            layer.update()


class ResNet(MiniFramework.NeuralNet):
    def __init__(self, params, block, num_blocks, num_classes=10, model_name=None):
        self.in_planes = 64
        self.hp = params
        super(ResNet, self).__init__(params, model_name)
        self.add_layer(ConLayer(32, 64, 3, stride=1, padding=1, hp=params), name="con1")
        self.add_layer(BatchNormalLayer(64), name='bn1')
        self.add_layer(ActivationLayer(ReLU()), name='relu1')
        # self.add_layer(PoolingLayer(kernel_size=3, stride=3, padding=1),name='pool')
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, layer_name="block1")
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, layer_name="block2")
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, layer_name="block3")
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, layer_name="block4")
        self.add_layers(self.layer1, name="block1")
        self.add_layers(self.layer2, name="block2")
        self.add_layers(self.layer3, name="block3")
        self.add_layers(self.layer4, name="block4")

    def _make_layer(self, block, planes, num_blocks, stride=1, layer_name=None):
        strides = [stride] + [1] * (num_blocks - 1)
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = {
                "_downsample_con": ConLayer(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride,
                                            hp=self.hp),
                "_downsample_bn": BatchNormalLayer(planes * block.expansion)}

        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.hp, layer_name, downsample))
            self.in_planes = planes * block.expansion
        return layers


if __name__ == "__main__":
    max_epoch = 5
    batch_size = 128
    learning_rate = 0.1
    params = HyperParameters(
        learning_rate, max_epoch, batch_size,
        net_type=NetType.MultipleClassifier,
        init_method=InitialMethod.Xavier,
        optimizer_name=OptimizerName.Momentum)

    net = ResNet(params=params, model_name="ResNet", block=BasicBlock, num_blocks=[2, 2, 2, 2])
