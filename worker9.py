from multiprocessing import managers, Queue, Value

from Model.vgg import *

cifar_name = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]


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
            train_x, train_y = load_CIFAR_batch(filename=f"./data/cifar-10-batches-py/{cifar_name[i]}")
        else:
            train_X, train_Y = load_CIFAR_batch(filename=f"./data/cifar-10-batches-py/{cifar_name[i]}")
            train_x = np.concatenate((train_x, train_X))
            train_y = np.concatenate((train_y, train_Y))
    test_x, test_y = load_CIFAR_batch(filename=f"./data/cifar-10-batches-py/test_batch")

    mdr = CIFAR10DataReader(train_x, train_y, test_x, test_y)
    # mdr = MnistDataReader(train_x,train_y,test_x,test_y)
    mdr.ReadData()
    mdr.NormalizeX()
    mdr.NormalizeY(NetType.MultipleClassifier, base=0)
    mdr.Shuffle()
    # mdr.distributed_Shuffle()
    mdr.GenerateValidationSet(k=12)
    return mdr


class queuemanager(managers.BaseManager):
    pass

order_queue = Queue()

def return_order_queue():
    global order_queue
    return order_queue


if __name__ == "__main__":
    datareader = LoadData()
    datareader.server_Shuffle()
    queuemanager.register("send_order")
    manager = queuemanager(address=("131.181.249.163", 10004), authkey=b"abc")
    manager.connect()
    order = manager.send_order()
    new_order = order.get()

    print(new_order)
    datareader.reorder(new_order)

