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

    def __init__(self, in_planes, planes, stride, param, layer_name):
        super().__init__(param, layer_name)
        self.add_layer(conv3x3(in_planes, planes, param=param, stride=stride), layer_name + "_con1")
        self.add_layer(BatchNormalLayer(planes), layer_name + "_bn1")
        self.add_layer(ReLU(), layer_name + "_relu1")
        self.add_layer(conv3x3(planes, planes, param=param), layer_name + "_con2")
        self.add_layer(BatchNormalLayer(planes), layer_name + "_bn2")
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = Shortcut(in_planes, planes, self.expansion, stride, param=param, layer_name=layer_name)
        else:
            self.shortcut = 0
        self.add_layer(Add(), layer_name + "_add")
        self.add_layer(ReLU(), layer_name + "_relu2")
        self.stride = stride

    def forward(self, input_v, train=True):
        output = None
        if isinstance(self.shortcut, Shortcut):
            residual = self.shortcut.forward(input_v, train)
        else:
            residual = input_v
        for i in range(self.layer_count):
            try:
                if isinstance(self.layer_list[i], Add):
                    output = self.layer_list[i].forward(input_v, residual, train)
                else:
                    output = self.layer_list[i].forward(input_v, train)
            except Exception as ex:
                print(ex)
            input_v = output
        self.output_v = output
        return self.output_v

    def backward(self, delta_in, idx):
        shortcut_delta_out = None
        delta_out = None
        # delta_in = self.output_v - Y
        for i in range(self.layer_count - 1, -1, -1):
            layer = self.layer_list[i]
            if isinstance(self.layer_list[i], Add):
                delta_out = layer.backward(delta_in, i)
                if isinstance(self.shortcut, Shortcut):
                    shortcut_delta_out = self.shortcut.backward(delta_out[1], i)
                else:
                    shortcut_delta_out = delta_out[1]
                delta_out = delta_out[0]
            else:
                delta_out = layer.backward(delta_in, i)
            delta_in = delta_out
        delta_out = delta_out + shortcut_delta_out
        return delta_out

    def update(self):
        for i in range(self.layer_count - 1, -1, -1):
            layer = self.layer_list[i]
            layer.update()
        if isinstance(self.shortcut, Shortcut):
            self.shortcut.update()


class BasicBlock1(MiniFramework.NeuralNet):
    expansion = 1

    def __init__(self, in_planes, planes, stride, param, layer_name, downsample=None):
        super().__init__(param, layer_name)
        self.add_layer(conv3x3(in_planes, planes, param=param, stride=stride), layer_name + "_con1")
        self.add_layer(BatchNormalLayer(planes), layer_name + "_bn1")
        self.add_layer(ReLU(), layer_name + "_relu1")
        self.add_layer(conv3x3(planes, planes, param=param), layer_name + "_con2")
        self.add_layer(BatchNormalLayer(planes), layer_name + "_bn2")
        self.shortcut = downsample
        self.add_layer(Add(), layer_name + "_add")
        self.add_layer(ReLU(), layer_name + "_relu2")
        self.stride = stride

    def forward(self, input_v, train=True):
        output = None
        if isinstance(self.shortcut, Shortcut):
            residual = self.shortcut.forward(input_v, train)
        else:
            residual = input_v
        for i in range(self.layer_count):
            try:
                if isinstance(self.layer_list[i], Add):
                    output = self.layer_list[i].forward(input_v, residual, train)
                else:
                    output = self.layer_list[i].forward(input_v, train)
            except Exception as ex:
                print(ex)
            input_v = output
        self.output_v = output
        return self.output_v

    def backward(self, delta_in, idx):
        shortcut_delta_out = None
        delta_out = None
        # delta_in = self.output_v - Y
        for i in range(self.layer_count - 1, -1, -1):
            layer = self.layer_list[i]
            if isinstance(self.layer_list[i], Add):
                delta_out = layer.backward(delta_in, i)
                if isinstance(self.shortcut, Shortcut):
                    shortcut_delta_out = self.shortcut.backward(delta_out[1], i)
                else:
                    shortcut_delta_out = delta_out[1]
                delta_out = delta_out[0]
            else:
                delta_out = layer.backward(delta_in, i)
            delta_in = delta_out
        delta_out = delta_out + shortcut_delta_out
        return delta_out

    def update(self):
        for i in range(self.layer_count - 1, -1, -1):
            layer = self.layer_list[i]
            layer.update()
        if isinstance(self.shortcut, Shortcut):
            self.shortcut.update()


class Bottleneck(NeuralNet):
    expansion = 1

    def __init__(self, in_planes, planes, stride, param, layer_name):
        super().__init__(param, layer_name)
        self.add_layer(conv3x3(in_planes, planes, param=param, stride=stride), layer_name + "_con1")
        self.add_layer(BatchNormalLayer(planes), layer_name + "_bn1")
        self.add_layer(ReLU(), layer_name + "_relu1")
        self.add_layer(conv3x3(planes, planes, param=param), layer_name + "_con2")
        self.add_layer(BatchNormalLayer(planes), layer_name + "_bn2")
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = Shortcut(in_planes, planes, self.expansion, stride, param=param, layer_name=layer_name)
        else:
            self.shortcut = 0
        self.add_layer(Add(), layer_name + "_add")
        self.add_layer(ReLU(), layer_name + "_relu2")
        self.stride = stride


class Shortcut(NeuralNet):
    def __init__(self, in_planes: int, planes: int, expansion: int, stride: int, param, layer_name):
        super().__init__(param, layer_name)
        self.add_layer(ConLayer(in_planes, planes * expansion, kernel_size=1, stride=stride,
                                hp=self.hp), name=layer_name + "_shortcut_con")
        self.add_layer(BatchNormalLayer(planes * expansion), name="_shortcut_bn")

    def forward(self, input_v, train):
        output = None
        for i in range(self.layer_count):
            try:
                output = self.layer_list[i].forward(input_v, train)
            except:
                print(i)
            input_v = output

        self.output_v = output
        return self.output_v

    def backward(self, delta_in, idx):

        for i in range(self.layer_count - 1, -1, -1):
            layer = self.layer_list[i]
            delta_out = layer.backward(delta_in, i)
            delta_in = delta_out
        delta_out = delta_in
        return delta_out

    def update(self):
        for i in range(self.layer_count - 1, -1, -1):
            layer = self.layer_list[i]
            layer.update()


class ResNet(MiniFramework.NeuralNet):
    def __init__(self, params, block, num_blocks, num_classes=10, model_name=None):
        self.in_planes = 64
        self.hp = params
        super(ResNet, self).__init__(params, model_name)
        self.add_layer(ConLayer(3, 64, 3, stride=1, padding=1, hp=params), name="con1")
        self.add_layer(BatchNormalLayer(64), name='bn1')
        self.add_layer(ReLU(), name='relu1')
        # self.add_layer(PoolingLayer(kernel_size=3, stride=3, padding=1),iteration_count='pool')
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, layer_name="block1")
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, layer_name="block2")
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, layer_name="block3")
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, layer_name="block4")
        self.add_layers(self.layer1, name="block1")
        self.add_layers(self.layer2, name="block2")
        self.add_layers(self.layer3, name="block3")
        self.add_layers(self.layer4, name="block4")
        self.add_layer(PoolingLayer(4, pooling_type=PoolingTypes.MEAN), name="avgpool")
        self.add_layer(FCLayer(512 * block.expansion, num_classes, hp=params), name="fc1")
        self.add_layer(Softmax(), name="softmax1")

    def _make_layer(self, block, planes, num_blocks, stride=1, layer_name=None):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.hp, layer_name + f"_{stride}"))
            self.in_planes = planes * block.expansion
        return layers

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

    # def __pre_update(self):
    #     for i in range(self.layer_count - 1, -1, -1):
    #         layer = self.layer_list[i]
    #         layer.pre_update()

    def update(self):
        for i in range(self.layer_count - 1, -1, -1):
            layer = self.layer_list[i]
            layer.update()


class Resnet_cifar10(NeuralNet):
    def __init__(self, params, block, num_blocks, num_classes=10, model_name=None):
        super(Resnet_cifar10, self).__init__(params, model_name)
        self.in_planes = 16
        self.add_layer(ConLayer(3, 16, 3, stride=1, padding=0, hp=params), name="con1")
        self.add_layer(BatchNormalLayer(16), name='bn1')
        self.add_layer(ReLU(), name='relu1')
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1, layer_name="block1")
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, layer_name="block2")
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, layer_name="block3")
        self.add_layers(self.layer1, name="block1")
        self.add_layers(self.layer2, name="block2")
        self.add_layers(self.layer3, name="block3")
        self.add_layer(PoolingLayer(8, pooling_type=PoolingTypes.MEAN), name="avgpool")
        self.add_layer(FCLayer(64, num_classes, hp=params), name="fc1")
        self.add_layer(Softmax(), name="softmax1")

    def _make_layer(self, block, planes, num_blocks, stride=1, layer_name=None):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.hp, layer_name))
            self.in_planes = planes * block.expansion
        return layers

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

    # def __pre_update(self):
    #     for i in range(self.layer_count - 1, -1, -1):
    #         layer = self.layer_list[i]
    #         layer.pre_update()

    def update(self):
        for i in range(self.layer_count - 1, -1, -1):
            layer = self.layer_list[i]
            layer.update()

    def _make_layers(self, block, planes, num_blocks, stride=1, layer_name=None):
        downsample = None
        if (stride != 1) or (self.in_planes != planes):
            downsample = Shortcut(self.in_planes, planes, stride=stride, param=self.hp, layer_name=layer_name)
        layers = [block(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes
        for i in range(1, num_blocks):
            layers.append(block(planes, planes))
        return layers


if __name__ == "__main__":
    max_epoch = 5
    batch_size = 128
    learning_rate = 0.1
    params = HyperParameters(learning_rate, max_epoch, batch_size, net_type=NetType.MultipleClassifier,
                             optimizer_name=OptimizerName.Momentum)

    net = ResNet(params=params, model_name="ResNet", block=BasicBlock, num_blocks=[2, 2, 2, 2])
    net1 = Resnet_cifar10(params=params, model_name="ResNet_cifar10", block=BasicBlock, num_blocks=[2, 2, 2])

    x = np.random.rand(2, 3, 32, 32)
    net1.forward(x)
    Y = np.random.randint(0, 10, [2, 10])
    net1.backward(Y)
    net1.update()
    print("resnet")
