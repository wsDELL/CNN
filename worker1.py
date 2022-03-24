import time
import multiprocessing
from multiprocessing.managers import BaseManager
from multiprocessing import Queue, Array

from MiniFramework import *
from Model.vgg import *

train_x = "./data/MNIST/raw/train-images-idx3-ubyte"
train_y = "./data/MNIST/raw/train-labels-idx1-ubyte"
test_x = "./data/MNIST/raw/t10k-images-idx3-ubyte"
test_y = "./data/MNIST/raw/t10k-labels-idx1-ubyte"

class QueueManager(BaseManager):
    pass


QueueManager.register('get_task_queue')
QueueManager.register('get_result_queue')

cifar_name = ["data_batch_1","data_batch_2","data_batch_3","data_batch_4","data_batch_5"]

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_CIFAR_batch(filename):
    # """ load single batch of cifar """
    # # with open(filename, 'rb') as f:
        datadict = unpickle(filename)  # dict类型
        X = datadict[b'data']  # X, ndarray, 像素值
        Y = datadict[b'labels']  # Y, list, 标签, 分类

        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y

def LoadData():
    for i in range(len(cifar_name)):
        if i == 0:
            train_x,train_y=load_CIFAR_batch(filename=f"./data/cifar-10-batches-py/{cifar_name[i]}")
        else:
            train_X,train_Y=load_CIFAR_batch(filename=f"./data/cifar-10-batches-py/{cifar_name[i]}")
            train_x = np.concatenate((train_x,train_X))
            train_y = np.concatenate((train_y,train_Y))
    test_x,test_y = load_CIFAR_batch(filename=f"./data/cifar-10-batches-py/test_batch")

    mdr = CIFAR10DataReader(train_x, train_y, test_x, test_y)
    # mdr = MnistDataReader(train_x,train_y,test_x,test_y)
    mdr.ReadData()
    mdr.NormalizeX()
    mdr.NormalizeY(NetType.MultipleClassifier, base=0)
    mdr.Shuffle()
    mdr.GenerateValidationSet(k=12)
    return mdr

def LoadData1():
    mdr = MnistDataReader(train_x, train_y, test_x, test_y, "image")
    mdr.ReadData()
    mdr.NormalizeX()
    mdr.NormalizeY(NetType.MultipleClassifier, base=0)
    mdr.Shuffle()
    mdr.GenerateValidationSet(k=12)
    return mdr


def model():
    num_output = 10
    max_epoch = 5
    batch_size = 128
    learning_rate = 0.1
    params = HyperParameters(
        learning_rate, max_epoch, batch_size,
        net_type=NetType.MultipleClassifier,
        init_method=InitialMethod.Xavier,
        optimizer_name=OptimizerName.Momentum)

    net = NeuralNet(params, "mnist_cnn")

    c1 = ConLayer(1, 8, kernel_size=3, hp=params, stride=1)
    net.add_layer(c1, "c1")
    r1 = ReLU()
    net.add_layer(r1, "relu1")
    p1 = PoolingLayer(kernel_size=2, stride=2, pooling_type=PoolingTypes.MAX)
    net.add_layer(p1, "p1")

    c2 = ConLayer(8, 16, kernel_size=3, hp=params, stride=1)
    net.add_layer(c2, "c2")
    r2 = ReLU()
    net.add_layer(r2, "relu2")
    p2 = PoolingLayer(kernel_size=2, stride=2, pooling_type=PoolingTypes.MAX)
    net.add_layer(p2, "p2")

    f3 = FCLayer(400, 32, params)
    net.add_layer(f3, "f3")
    bn3 = BatchNormalLayer(32)
    net.add_layer(bn3, "bn3")
    r3 = ReLU()
    net.add_layer(r3, "relu3")

    f4 = FCLayer(32, 10, params)
    net.add_layer(f4, "f2")
    s4 = Softmax()
    net.add_layer(s4, "s4")


    # c1 = ConLayer((1, 28, 28), (8, 3, 3), params, stride=1, pad=0)
    # net.add_layer(c1, "c1")
    # r1 = ActivationLayer(ReLU())
    # net.add_layer(r1, "relu1")
    # p1 = PoolingLayer(c1.output_shape, (2, 2), 2, PoolingTypes.MAX)
    # net.add_layer(p1, "p1")
    #
    # c2 = ConLayer(p1.output_shape, (16, 3, 3), params, stride=1, pad=0)
    # net.add_layer(c2, "c2")
    # r2 = ActivationLayer(ReLU())
    # net.add_layer(r2, "relu2")
    # p2 = PoolingLayer(c2.output_shape, (2, 2), 2, PoolingTypes.MAX)
    # net.add_layer(p2, "p2")
    #
    # f3 = FCLayer(p2.output_size, 32, params)
    # net.add_layer(f3, "f3")
    # bn3 = BatchNormalLayer(f3.output_num)
    # net.add_layer(bn3, "bn3")
    # r3 = ActivationLayer(ReLU())
    # net.add_layer(r3, "relu3")
    #
    # f4 = FCLayer(f3.output_num, 10, params)
    # net.add_layer(f4, "f2")
    # s4 = ClassificationLayer(Softmax())
    # net.add_layer(s4, "s4")

    return net


def model1():
    num_output = 10
    max_epoch = 20
    batch_size = 128
    learning_rate = 0.1
    params = HyperParameters(learning_rate, max_epoch, batch_size,
                             net_type=NetType.MultipleClassifier,
                             init_method=InitialMethod.Kaiming_Uniform,
                             optimizer_name=OptimizerName.Adam)

    net = NeuralNet(params,"alexnet")

    c1 = ConLayer((1, 28, 28), (32, 3, 3), params, stride=1, pad=1)
    net.add_layer(c1, "c1")
    p1 = PoolingLayer(c1.output_shape, (2, 2), 2, PoolingTypes.MAX)
    net.add_layer(p1, "p1")
    r1 = ActivationLayer(ReLU())
    net.add_layer(r1, "relu1")

    c2 = ConLayer(p1.output_shape, (64, 3, 3), params, stride=1, pad=1)
    net.add_layer(c2, "c2")
    p2 = PoolingLayer(c2.output_shape, (2, 2), 2, PoolingTypes.MAX)
    net.add_layer(p2, "p2")
    r2 = ActivationLayer(ReLU())
    net.add_layer(r2, "relu2")

    c3 = ConLayer(p2.output_shape, (128, 3, 3), params, stride=1, pad=1)
    net.add_layer(c3, "c3")
    # r3 = ActivationLayer(ReLU())
    # net.add_layer(r3, "relu3")

    c4 = ConLayer(c3.output_shape, (256, 3, 3), params, stride=1, pad=1)
    net.add_layer(c4, "c3")
    # r4 = ActivationLayer(ReLU())
    # net.add_layer(r4, "relu4")

    c5 = ConLayer(c4.output_shape, (256, 3, 3), params, stride=1, pad=1)
    net.add_layer(c5, "c3")
    p5 = PoolingLayer(c5.output_shape, (3, 3), 2, PoolingTypes.MAX)
    net.add_layer(p5, "p2")
    r5 = ActivationLayer(ReLU())
    net.add_layer(r5, "relu5")

    # d1 = DropoutLayer(p5.output_shape)
    # net.add_layer(d1, 'd1')
    f1 = FCLayer(256 * 3 * 3, 1024, params)
    net.add_layer(f1, "f1")
    bn1 = BatchNormalLayer(f1.output_num)
    net.add_layer(bn1, 'bn1')
    r6 = ActivationLayer(ReLU())
    net.add_layer(r6, "relu6")

    # d1 = DropoutLayer(f1.output_num)
    # net.add_layer(d1, "d1")

    f2 = FCLayer(f1.output_num, 512, params)
    net.add_layer(f2, "f2")
    bn2 = BatchNormalLayer(f2.output_num)
    net.add_layer(bn2, 'bn1')
    r6 = ActivationLayer(ReLU())
    net.add_layer(r6, "relu6")

    f3 = FCLayer(f2.output_num, 10, params)
    net.add_layer(f3, "f3")
    s4 = ClassificationLayer(Softmax())
    net.add_layer(s4, "s4")

    return net


if __name__ == '__main__':
    dataReader = LoadData()
    # net = model()
    num_output = 10
    max_epoch = 75
    batch_size = 128
    learning_rate = 0.01
    params = HyperParameters(
        learning_rate, max_epoch, batch_size,
        net_type=NetType.MultipleClassifier,
        init_method=InitialMethod.Kaiming_Uniform,
        optimizer_name=OptimizerName.Adam)
    net = VGG(param=params,vgg_name="VGG11")
    server_address = '131.181.249.163'
    print('connect to server %s...' % server_address)

    manager = QueueManager(address=(server_address, 5006), authkey=b'abc')
    manager.connect()

    task = manager.get_task_queue()
    result = manager.get_result_queue()

    while True:
        count = 0
        if task.empty():
            time.sleep(0.1)
            # count = count + 1
            # if count > 500:
            #     break
        n = task.get()
        iteration = n
        name = list(iteration.keys())[0]
        net.distributed_load_parameters(iteration)
        batch_x, batch_y = dataReader.GetBatchTrainSamples(net.hp.batch_size, name)
        print(f'worker get {name}')
        param = net.distributed_train(batch_x,batch_y)
        iteration[name] = param
        result.put(iteration)

    print('worker exit')
