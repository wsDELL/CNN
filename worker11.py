import time
import multiprocessing
from multiprocessing.managers import BaseManager
from multiprocessing import Queue, Array

from MiniFramework import *
from Model.alexnet import AlexNet
from Model.vgg import *

train_x = "./data/MNIST/raw/train-images-idx3-ubyte"
train_y = "./data/MNIST/raw/train-labels-idx1-ubyte"
test_x = "./data/MNIST/raw/t10k-images-idx3-ubyte"
test_y = "./data/MNIST/raw/t10k-labels-idx1-ubyte"

class QueueManager(BaseManager):
    pass


QueueManager.register('get_task_queue')
QueueManager.register('get_gradient_queue')
QueueManager.register('send_order_queue')
QueueManager.register('send_training_order_queue')


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
    return mdr

def LoadData1():
    mdr = MnistDataReader(train_x, train_y, test_x, test_y, "image")
    mdr.ReadData()
    mdr.NormalizeX()
    mdr.NormalizeY(NetType.MultipleClassifier, base=0)
    mdr.Shuffle()
    mdr.GenerateValidationSet(k=12)
    return mdr

def dis_Alexnet():
    num_output = 10
    max_epoch = 40
    batch_size = 128
    learning_rate = 0.01
    params = HyperParameters(learning_rate, max_epoch, batch_size, net_type=NetType.MultipleClassifier,
                             optimizer_name=OptimizerName.Adam, regular_name=RegularMethod.L2, regular_value=0.0005)

    net = NeuralNet(params, "dis_alexnet")

    c1 = ConLayer(3, 64, kernel_size=3, hp=params, stride=2, padding=1)
    net.add_layer(c1, "c1")
    r1 = ReLU()
    net.add_layer(r1, "relu1")
    p1 = PoolingLayer(kernel_size=2, stride=2, pooling_type=PoolingTypes.MAX)
    net.add_layer(p1, "p1")

    c2 = ConLayer(64, 192, kernel_size=3, hp=params, stride=1, padding=1)
    net.add_layer(c2, "c2")
    r2 = ReLU()
    net.add_layer(r2, "relu2")
    p2 = PoolingLayer(kernel_size=2, stride=2, pooling_type=PoolingTypes.MAX)
    net.add_layer(p2, "p2")

    c3 = ConLayer(192, 384, kernel_size=3, hp=params, stride=1, padding=1)
    net.add_layer(c3, "c3")
    r3 = ReLU()
    net.add_layer(r3, "relu3")

    c4 = ConLayer(384, 256, kernel_size=3, hp=params, stride=1, padding=1)
    net.add_layer(c4, "c4")
    r4 = ReLU()
    net.add_layer(r4, "relu4")

    c5 = ConLayer(256, 256, kernel_size=3, hp=params, stride=1, padding=1)
    net.add_layer(c5, "c5")
    r5 = ReLU()
    net.add_layer(r5, "relu5")
    p5 = PoolingLayer(kernel_size=2, stride=2, pooling_type=PoolingTypes.MEAN)
    net.add_layer(p5, "p5")

    d1 = DropoutLayer(ratio=0.3)
    net.add_layer(d1, 'd1')
    f1 = FCLayer(256 * 2 * 2, 1024, params)
    net.add_layer(f1, "f1")
    bn1 = BatchNormalLayer(f1.output_num)
    net.add_layer(bn1, 'bn1')
    r6 = ReLU()
    net.add_layer(r6, "relu6")

    d2 = DropoutLayer(ratio=0.3)
    net.add_layer(d2, "d2")
    f2 = FCLayer(1024, 1024, params)
    net.add_layer(f2, "f2")
    bn2 = BatchNormalLayer(f2.output_num)
    net.add_layer(bn2, 'bn2')
    r6 = ReLU()
    net.add_layer(r6, "relu6")

    f3 = FCLayer(f2.output_num, 10, params)
    net.add_layer(f3, "f3")
    s4 = Softmax()
    net.add_layer(s4, "s4")

    return net


if __name__ == '__main__':
    # dataReader = LoadData()
    num_output = 10
    max_epoch = 50
    batch_size = 128
    learning_rate = 0.01
    params = HyperParameters(learning_rate, max_epoch, batch_size, net_type=NetType.MultipleClassifier,
                             optimizer_name=OptimizerName.Adam, regular_name=RegularMethod.L2, regular_value=0.0005)
    # net = VGG(param=params,vgg_name="VGG11")
    net = AlexNet(param=params, model_name="dis_Alexnet")

    server_address = '131.181.249.163'
    print('connect to server %s...' % server_address)

    manager = QueueManager(address=(server_address, 5006), authkey=b'abc')
    while True:
        try:
            manager.connect()
            break
        except:
            print("no connection")
            time.sleep(5)

    total_order = manager.send_order_queue()
    training_order = manager.send_training_order_queue()
    task = manager.get_task_queue()
    result = manager.get_gradient_queue()
    dataReader = LoadData()
    if not total_order.empty():
        data_order = total_order.get()
        dataReader.ReadData()
        dataReader.total_reorder(data_order)
        dataReader.GenerateValidationSet(k=12)
        dataReader.NormalizeX()
        dataReader.NormalizeY(NetType.MultipleClassifier, base=0)
    while True:
        count = 0
        if task.empty():
            time.sleep(0.1)
            # count = count + 1
            # if count > 500:
            #     break
        if not training_order.empty():
            data_order = training_order.get()
            dataReader.training_reorder(data_order)
            print("Shuffle finish")
        iteration = task.get()
        iteration_count = list(iteration.keys())[0]
        net.distributed_load_parameters(iteration)
        batch_x, batch_y = dataReader.GetBatchTrainSamples(net.hp.batch_size, iteration_count)
        print(f'worker get {iteration_count}')
        grad = net.distributed_train(batch_x,batch_y)
        iteration[iteration_count] = grad
        result.put(iteration)

    print('worker exit')
