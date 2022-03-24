import numpy as np

from MiniFramework.DataReader import *
import struct


class MnistDataReader(DataReader):
    def __init__(self, train_x, train_y, test_x, test_y, mode="image"):
        self.train_image_file = train_x
        self.train_label_file = train_y
        self.test_image_file = test_x
        self.test_label_file = test_y
        self.num_example = 0
        self.num_feature = 0
        self.num_category = 0
        self.num_validation = 0
        self.num_train = 0
        self.num_test = 0
        self.mode = mode

    def ReadAllData(self, count):
        self.XTrainRaw = self.ReadImageFile(self.train_image_file)
        self.YTrainRaw = self.ReadImageFile(self.train_label_file)
        self.XTestRaw = self.ReadImageFile(self.test_image_file)
        self.YTestRaw = self.ReadImageFile(self.test_label_file)
        self.XTrainRaw = self.XTrainRaw[0:count]
        self.YTrainRaw = self.YTrainRaw[0:count]
        self.num_example = self.XTrainRaw.shape[0]
        self.num_category = (np.unique(self.YTrainRaw)).shape[0]
        self.num_test = self.XTestRaw.shape[0]
        self.num_train = self.num_example
        if self.mode == "vector":
            self.num_feature = 784
        self.num_validation = 0
        
    def ReadData(self):
        self.XTrainRaw = self.ReadImageFile(self.train_image_file)
        self.YTrainRaw = self.ReadLabelFile(self.train_label_file)
        self.XTestRaw = self.ReadImageFile(self.test_image_file)
        self.YTestRaw = self.ReadLabelFile(self.test_label_file)
        self.num_example = self.XTrainRaw.shape[0]
        self.num_category = (np.unique(self.YTrainRaw)).shape[0]
        self.num_test = self.XTestRaw.shape[0]
        self.num_train = self.num_example
        if self.mode == "vector":
            self.num_feature = 784
        self.num_validation = 0

    def ReadImageFile(self, image_file_name):
        with open(image_file_name, 'rb') as f:
            fb_data = f.read()
        f.close()
        fmt_header = '>iiii'
        offset = 0
        magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, fb_data, offset)
        image_size = num_rows * num_cols
        offset += struct.calcsize(fmt_header)
        fmt_image = '>' + str(image_size) + 'B'
        images = np.empty((num_images, 1, num_rows, num_cols))
        for i in range(num_images):
            im = struct.unpack_from(fmt_image, fb_data, offset)
            images[i] = np.array(im).reshape((1, num_rows, num_cols))
            offset += struct.calcsize(fmt_image)

        return images

    def ReadLabelFile(self, train_label_name):
        with open(train_label_name, 'rb') as f:
            fb_data = f.read()

        offset = 0
        fmt_header = '>ii'
        magic_number, label_num = struct.unpack_from(fmt_header, fb_data, offset)
        offset += struct.calcsize(fmt_header)
        labels = []
        fmt_label = '>B'
        for i in range(label_num):
            labels.append(struct.unpack_from(fmt_label, fb_data, offset)[0])
            offset += struct.calcsize(fmt_label)
        labels = np.array(labels)
        labels = labels[:,np.newaxis]
        return labels

    def NormalizeX(self):
        self.XTrain = self.__NormalizeData(self.XTrainRaw)
        self.XTest = self.__NormalizeData(self.XTestRaw)

    def __NormalizeData(self, XRawData):
        X_New = np.zeros(XRawData.shape).astype('float32')
        x_max = np.max(XRawData)
        x_min = np.min(XRawData)
        X_New = (XRawData - x_min)/(x_max - x_min)
        return X_New

    def GetBatchTrainSamples(self, batch_size, iteration):
        start = iteration * batch_size
        end = start +batch_size
        if self.num_validation == 0:
            batch_X = self.XTrain[start:end]
            batch_Y = self.YTrain[start:end]
        else:
            batch_X = self.XTrain[start:end]
            batch_Y = self.YTrain[start:end]
        if self.mode == "vector":
            return batch_X.reshape(-1, 784), batch_Y
        elif self.mode == "image":
            return batch_X, batch_Y

    def GetValidationSet(self):
        batch_X = self.XDev
        batch_Y = self.YDev
        if self.mode == "vector":
            return batch_X.reshape(self.num_validation, -1), batch_Y
        elif self.mode == "image":
            return batch_X, batch_Y

    def GetTestSet(self):
        if self.mode == "vector":
            return self.XTest.reshape(self.num_test,-1), self.YTest
        elif self.mode == "image":
            return self.XTest, self.YTest

    def Shuffle(self):
        seed = np.random.randint(0, 100)
        np.random.seed(seed)
        XP = np.random.permutation(self.XTrain)
        np.random.seed(seed)
        YP = np.random.permutation(self.YTrain)
        self.XTrain = XP
        self.YTrain = YP
        return self.XTrain, self.YTrain

    def data_Shuffle(self):
        seed = random.randint(0, 100)
        random.seed(seed)
        order = [i for i in range(len(self.XTrain))]
        random.shuffle(order)
        return order

    def reorder(self,new_order):
        XP = self.XTrain[new_order,:,:,:]
        YP = self.YTrain[new_order,:]
        self.XTrain = XP
        self.YTrain = YP




