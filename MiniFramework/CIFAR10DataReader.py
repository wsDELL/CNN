import numpy as np

from MiniFramework.util import *
from MiniFramework.Enums import *
import random


class CIFAR10DataReader(object):
    def __init__(self, x_train, y_train, x_test, y_test):
        # self.train_file_name = train_file
        # self.test_file_name = test_file
        self.num_train = 0  # num of training examples
        self.num_test = 0  # num of test examples
        self.num_validation = 0  # num of validation examples
        self.num_feature = 0  # num of features
        self.num_category = 0  # num of categories
        self.XTrain = None  # training feature set
        self.YTrain = None  # training label set
        self.XTest = None  # test feature set
        self.YTest = None  # test label set
        self.XTrainRaw = np.transpose(x_train, axes=(0, 3, 1, 2))  # training feature set before normalization
        self.YTrainRaw = y_train  # training label set before normalization
        self.XTestRaw = np.transpose(x_test, axes=(0, 3, 1, 2))  # test feature set before normalization
        self.YTestRaw = y_test  # test label set before normalization
        self.XDev = None  # validation feature set
        self.YDev = None  # validation lable set
        self.order = None

    def ReadData(self):
        self.XTrainRaw = self.XTrainRaw.astype('float32')
        self.YTrainRaw = self.YTrainRaw.astype('int32')
        self.YTrainRaw = self.YTrainRaw.reshape(self.YTrainRaw.size, 1)
        assert (self.XTrainRaw.shape[0] == self.YTrainRaw.shape[0])
        self.num_train = self.XTrainRaw.shape[0]
        self.num_feature = self.XTrainRaw.shape[1]
        self.num_category = len(np.unique(self.YTrainRaw))
        self.XTrain = self.XTrainRaw
        self.YTrain = self.YTrainRaw
        self.XTestRaw = self.XTestRaw.astype('float32')
        self.YTestRaw = self.YTestRaw.astype('int32')
        self.YTestRaw = self.YTestRaw.reshape(self.YTestRaw.size, 1)
        assert (self.XTestRaw.shape[0] == self.YTestRaw.shape[0])
        self.XTest = self.XTestRaw
        self.YTest = self.YTestRaw
        self.XDev = self.XTest
        self.YDev = self.YTest

    # else:
    #     raise Exception("No test file")

    def NormalizeX(self):
        # self.XTrain = self.__NormalizeData(self.XTrainRaw)
        self.XTrain = self.__NormalizeData(self.XTrain)
        self.XTest = self.__NormalizeData(self.XTestRaw)
        self.XDev = self.__NormalizeData(self.XDev)

    def __NormalizeData(self, XRawData):
        X_New = np.zeros(XRawData.shape).astype('float32')
        x_max = np.max(XRawData)
        x_min = np.min(XRawData)
        X_New = (XRawData - x_min) / (x_max - x_min)
        return X_New

    def NormalizeY(self, nettype, base=0):
        if nettype == NetType.Fitting:
            y_merge = np.vstack((self.YTrainRaw, self.YTestRaw))
            y_merge_norm = self.__NormalizeY(y_merge)
            train_count = self.YTrainRaw.shape[0]
            self.YTrain = y_merge_norm[0:train_count, :]
            self.YTest = y_merge_norm[train_count:, :]
        elif nettype == NetType.BinaryClassifier:
            self.YTrain = self.__ToZeroOne(self.YTrainRaw, base)
            self.YTest = self.__ToZeroOne(self.YTestRaw, base)
        elif nettype == NetType.MultipleClassifier:
            self.YDev = self.__ToOneHot(self.YDev, base)
            self.YTrain = self.__ToOneHot(self.YTrain, base)
            self.YTest = self.__ToOneHot(self.YTestRaw, base)

    def __NormalizeY(self, raw_data):
        assert (raw_data.shape[1] == 1)
        self.Y_norm = np.zeros((2, 1)).astype('float32')
        max_value = np.max(raw_data)
        min_value = np.min(raw_data)
        self.Y_norm[0, 0] = min_value
        self.Y_norm[1, 0] = max_value - min_value
        y_new = (raw_data - min_value) / self.Y_norm[1, 0]
        return y_new

    def DeNormalizeY(self, predict_data):
        real_value = predict_data * self.Y_norm[1, 0] + self.Y_norm[0, 0]
        return real_value

    def __ToOneHot(self, Y, base=0):
        count = Y.shape[0]
        temp_Y = np.zeros((count, self.num_category)).astype('float32')
        for i in range(count):
            n = int(Y[i, 0])
            temp_Y[i, n - base] = 1
        return temp_Y

    # for binary classifier
    # if use tanh function, need to set negative_value = -1
    def __ToZeroOne(Y, positive_label=1, negative_label=0, positiva_value=1, negative_value=0):
        temp_Y = np.zeros_like(Y).astype('float32')
        for i in range():
            if Y[i, 0] == negative_label:
                temp_Y[i, 0] = negative_value
            elif Y[i, 0] == positive_label:
                temp_Y[i, 0] = positiva_value
            # end if
        # end for
        return temp_Y

    # normalize data by specified range and min_value
    def NormalizePredicateData(self, X_predicate):
        X_new = np.zeros(X_predicate.shape).astype('float32')
        n_feature = X_predicate.shape[0]
        for i in range(n_feature):
            x = X_predicate[i, :]
            X_new[i, :] = (x - self.X_norm[0, i]) / self.X_norm[1, i]
        return X_new

    # need explicitly call this function to generate validation set
    def GenerateValidationSet(self, k=10):
        self.num_validation = int(self.num_train / k)
        self.num_train = self.num_train - self.num_validation
        # validation set
        self.XDev = self.XTrain[0:self.num_validation]
        self.YDev = self.YTrain[0:self.num_validation]
        # train set
        self.XTrain = self.XTrain[self.num_validation:]
        self.YTrain = self.YTrain[self.num_validation:]

    def GetValidationSet(self):
        return self.XDev, self.YDev

    def GetTestSet(self):
        return self.XTest, self.YTest

    # achieve batch data set
    def GetBatchTrainSamples(self, batch_size, iteration):
        start = iteration * batch_size
        # if start > self.num_train:
        #     iteration = iteration - 1
        # start = iteration * batch_size
        end = start + batch_size
        # if self.num_train - end <0:
        #     end = self.num_train
        batch_X = self.XTrain[start:end, :]
        batch_Y = self.YTrain[start:end, :]
        return batch_X, batch_Y

    # permutation only affect along the first axis, so we need transpose the array first
    # see the comment of this class to understand the data format
    def Shuffle(self):
        seed = np.random.randint(0, 100)
        np.random.seed(seed)
        order = np.random.permutation(len(self.XTrain))
        XP = self.XTrain[order, :, :, :]
        YP = self.YTrain[order, :]
        self.XTrain = XP
        self.YTrain = YP

    def data_Shuffle(self):
        seed = random.randint(0, 100)
        random.seed(seed)
        order = [i for i in range(len(self.XTrain))]
        random.shuffle(order)
        self.order = order

    def reorder(self,new_order):
        XP = self.XTrain[new_order,:,:,:]
        YP = self.YTrain[new_order,:]
        self.XTrain = XP
        self.YTrain = YP
