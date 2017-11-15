import numpy as np
import struct
import _init_
import matplotlib.pyplot as plt
from DeepLearning.deep_learning import Batch_Normalization
import cv2
import pickle as p
from _init_ import FLAGS

# -------------------------  mnist -------------------------
CHAR = "0123456789"


def load_CIFAR_batch(filename):
  """ load single batch of cifar """
  with open(filename, 'rb') as f:
    datadict = p.load(f, encoding='latin1')
    X = datadict['data']
    Y = datadict['labels']
    X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
    Y = np.array(Y)
    return X, Y


def loadImageSet(binfile):
    buffers = binfile.read()
    head = struct.unpack_from('>IIII', buffers, 0)
    offset = struct.calcsize('>IIII')
    imgNum = head[1]
    width = head[2]
    height = head[3]
    # [60000]*28*28
    bits = imgNum * width * height
    bitsString = '>' + str(bits) + 'B'  # like '>47040000B'
    imgs = struct.unpack_from(bitsString, buffers, offset)
    binfile.close()
    imgs = np.reshape(imgs, [imgNum, 1, width * height])
    return imgs


def loadLabelSet(binfile):
    buffers = binfile.read()
    head = struct.unpack_from('>II', buffers, 0)
    imgNum = head[1]
    offset = struct.calcsize('>II')
    numString = '>' + str(imgNum) + "B"
    labels = struct.unpack_from(numString, buffers, offset)
    binfile.close()
    labels = np.reshape(labels, [imgNum, 1])
    return labels


class mnist:
    def __init__(self):
        self.train_images_in = open("G:\\MNIST\\MNIST_data\\train-images.idx3-ubyte", 'rb')
        self.train_labels_in = open("G:\\MNIST\\MNIST_data\\train-labels.idx1-ubyte", 'rb')
        self.test_images_in = open("G:\\MNIST\\MNIST_data\\t10k-images.idx3-ubyte", 'rb')
        self.test_labels_in = open("G:\\MNIST\\MNIST_data\\t10k-labels.idx1-ubyte", 'rb')
        self.batch_size = FLAGS.batch_size
        self.train_image = loadImageSet(self.train_images_in)  # [60000, 1, 784]
        self.train_labels = loadLabelSet(self.train_labels_in)  # [60000, 1]
        self.test_images = loadImageSet(self.test_images_in)  # [10000, 1, 784]
        self.test_labels = loadLabelSet(self.test_labels_in)  # [10000, 1]
        self.data = {"train": self.train_image, "test": self.test_images}
        self.label = {"train": self.train_labels, "test": self.test_labels}
        self.indexes = {"train": 0, "val": 0, "test": 0}

    def get_mini_batch(self, data_name="train"):
        if (self.indexes[data_name] + 1) * self.batch_size > self.data[data_name].shape[0]:
            self.indexes[data_name] = 0
        batch_data = self.data[data_name][
                     self.indexes[data_name] * self.batch_size:(self.indexes[data_name] + 1) * self.batch_size, :, :]
        batch_label = self.label[data_name][
                      self.indexes[data_name] * self.batch_size:(self.indexes[data_name] + 1) * self.batch_size, :]
        self.indexes[data_name] += 1
        y = np.zeros((self.batch_size, len(CHAR)))
        for kk in range(self.batch_size):
            y[kk, CHAR.index(str(int(batch_label[kk])))] = 1.0
        x = Batch_Normalization(batch_data)
        x = np.reshape(x, (16, 784, 1))
        x = np.reshape(x, (16, 28, 28, 1))
        return x, y


# ---------------------  CIFAR10  ---------------------------------
"""
每个文件都是# image:(50000, 32, 32, 3) label:(50000,)大小，image是0-255没有做归一化的数据，label是0.0-9.0的数字
"G:\\cifar10\\data\\CIFAR10_train_image.npy"      # [50000, 32, 32, 3]
"G:\\cifar10\\data\\CIFAR10_train_label.npy"      # [50000]
"G:\\cifar10\\data\\CIFAR10_test_image.npy"       # [10000, 32, 32, 3]
"G:\\cifar10\\data\\CIFAR10_test_label.npy"       # [10000]
"""


class cifar10:
    def __init__(self):
        x1, y1 = load_CIFAR_batch("G:\\cifar10\\data\\data_batch_1")
        x2, y2 = load_CIFAR_batch("G:\\cifar10\\data\\data_batch_2")
        x3, y3 = load_CIFAR_batch("G:\\cifar10\\data\\data_batch_3")
        x4, y4 = load_CIFAR_batch("G:\\cifar10\\data\\data_batch_4")
        x5, y5 = load_CIFAR_batch("G:\\cifar10\\data\\data_batch_5")
        self.train_image = np.concatenate((x1, x2, x3, x4, x5), 0)
        self.train_labels = np.concatenate((y1, y2, y3, y4, y5))
        print(self.train_image.shape, self.train_labels.shape)
        # self.train_image = np.load("G:\\cifar10\\data\\CIFAR10_train_image.npy")  # [50000, 32, 32, 3]
        # self.train_labels = np.load("G:\\cifar10\\data\\CIFAR10_train_label.npy")  # [50000]
        self.test_image, self.test_labels = load_CIFAR_batch("G:\\cifar10\\data\\test_batch")
        # self.test_image = np.load("G:\\cifar10\\data\\CIFAR10_test_image.npy")  # [10000, 32, 32, 3]
        # self.test_labels = np.load("G:\\cifar10\\data\\CIFAR10_test_label.npy")  # [10000]
        self.data = {"train": self.train_image, "test": self.test_image}
        self.label = {"train": self.train_labels, "test": self.test_labels}
        self.indexes = {"train": 0, "val": 0, "test": 0}
        self.batch_size = FLAGS.batch_size
        self.shuffle = True
        self.remain = {"train": 0, "val": 0, "test": 0}

    def get_mini_batch(self, data_name="train"):
        # print(data_name, self.indexes[data_name], self.data[data_name].shape[0])
        if (self.indexes[data_name] + 1) * self.batch_size + self.remain[data_name] > self.data[data_name].shape[0]:
            remain_num = self.data[data_name].shape[0] - self.indexes[data_name] * self.batch_size - self.remain[data_name]
            # print("remain_num", remain_num)
            if remain_num != 0:
                batch_data_1 = self.data[data_name][-remain_num:, ...]
                batch_label_1 = self.label[data_name][-remain_num:]
                # print('Shuffling')
                order = np.random.permutation(self.data[data_name].shape[0])
                self.data[data_name] = self.data[data_name][order, ...]
                self.label[data_name] = self.label[data_name][order]
                batch_data_2 = self.data[data_name][0:FLAGS.batch_size-remain_num, ...]
                batch_label_2 = self.label[data_name][0:FLAGS.batch_size-remain_num]
                batch_data = np.concatenate((batch_data_1, batch_data_2), 0)
                batch_label = np.concatenate((batch_label_1, batch_label_2))
                # print("shape", batch_data.shape, batch_label.shape)
                self.indexes[data_name] = -1
                self.remain[data_name] = FLAGS.batch_size - remain_num
            else:
                if self.shuffle is True:
                    # print('Shuffling')
                    order = np.random.permutation(self.data[data_name].shape[0])
                    self.data[data_name] = self.data[data_name][order, ...]
                    self.label[data_name] = self.label[data_name][order]
                self.indexes[data_name] = 0
                batch_data = self.data[data_name][
                             self.indexes[data_name] * self.batch_size:(self.indexes[data_name] + 1) * self.batch_size,
                             :, :, :]
                batch_label = self.label[data_name][
                              self.indexes[data_name] * self.batch_size:(self.indexes[data_name] + 1) * self.batch_size]
        else:
            batch_data = self.data[data_name][
                         self.indexes[data_name] * self.batch_size + self.remain[data_name]:(self.indexes[data_name] + 1) * self.batch_size + self.remain[data_name], :, :, :]
            batch_label = self.label[data_name][
                          self.indexes[data_name] * self.batch_size + self.remain[data_name]:(self.indexes[data_name] + 1) * self.batch_size + self.remain[data_name]]
        self.indexes[data_name] += 1
        y = np.zeros((self.batch_size, len(CHAR)))
        for kk in range(self.batch_size):
            y[kk, CHAR.index(str(int(batch_label[kk])))] = 1.0
        if data_name == "train":
            data_argument = np.zeros((FLAGS.batch_size, 32+4*2, 32+4*2, 3))
            data_argument[:, 4:36, 4:36, :] = batch_data
            rand_data = np.zeros_like(batch_data)
            for ii in range(FLAGS.batch_size):
                x_begin = np.random.randint(0, 8)
                y_begin = np.random.randint(0, 8)
                argument_img = data_argument[ii, x_begin:x_begin+32, y_begin:y_begin+32, :]
                # 50%可能性翻转  沿y轴水平旋转
                flip_prop = np.random.randint(low=0, high=2)
                if flip_prop == 0:
                    argument_img = cv2.flip(argument_img, 1)
                rand_data[ii, :, :, :] = argument_img
            batch_data = rand_data

        # x = batch_data
        x = Batch_Normalization(batch_data)
        return x, y

#
# def plot_images(images, labels):
#     for i in np.arange(0, 16):
#         plt.subplot(4, 4, i + 1)
#         plt.axis('off')
#         plt.title(labels[i], fontsize=14)
#         plt.subplots_adjust(top=1.5)
#         plt.imshow(images[i], cmap='gray')
#     plt.show()
#
# if __name__ == "__main__":
#     cifar10 = cifar10
#     x, y = cifar10().get_mini_batch(data_name="train")
#     print(x.shape, y.shape)
#     plot_images(x, y)                           # [16, 1, 784]  [16, 1]
