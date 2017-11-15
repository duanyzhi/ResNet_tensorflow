from DeepLearning.deep_learning import one_hot, Batch_Normalization
from DeepLearning.Image import plot_images
from PIL import Image
import _init_
import numpy as np
import random
import cv2
import os


class get_images:
    def __init__(self):
        self.lines = _init_.classes_name
        self.classes_train = []
        self.classes_test = []
        for line in self.lines:
            count = 0
            path = _init_.cwd + line[:9] + "\\"
            assert len(os.listdir(path)) == 1300  # 判断文件夹文件个数是不是1300个
            for im_name in os.listdir(path):
                if count < 1000:
                    self.classes_train.append(im_name)
                else:
                    self.classes_test.append(im_name)
                count += 1
        random.shuffle(self.classes_train)
        random.shuffle(self.classes_test)
        self.data_images = np.zeros(
            (_init_.batch_size, _init_.input_image[0], _init_.input_image[1], _init_.input_image[2]))
        self.data_label = np.zeros(_init_.batch_size)
        self.count_train = 0
        self.count_test = 0
        self.len_train = len(self.classes_train)
        self.len_test = len(self.classes_test)

    def get_mini_batch(self, train=True):
        if train:
            if (self.count_train + 1) * _init_.batch_size < self.len_train:
                pass
            else:
                self.count_train = 0
            images_train, label_train = self.call_data(self.count_train, self.classes_train)
            self.count_train += 1
            return images_train, label_train
        else:
            if (self.count_test + 1) * _init_.batch_size < self.len_test:
                pass
            else:
                self.count_test = 0
            images_test, label_test = self.call_data(self.count_test, self.classes_test)
            self.count_test += 1
            return images_test, label_test

    def call_data(self, count, data_name):
        batch_name = data_name[count * _init_.batch_size:(count + 1) * _init_.batch_size]
        for index, img_name in enumerate(batch_name):
            img_path = _init_.cwd + img_name[:9] + "\\" + img_name
            img = cv2.imread(img_path)
            # 这里虽然img会是None，但是不会报错 可以继续，只有后面用的时候才会报错 所以这里判断一下是None立刻报错
            if img is None:
                raise ValueError
            try:
                self.data_images[index, :, :, :] = cv2.resize(img, (_init_.input_image[0], _init_.input_image[1]))
                self.data_label[index] = self.lines.index(img_name[:9])
            except ValueError:
                assert index != 0
                self.data_images[index, :, :, :] = self.data_images[0, :, :, :]
                self.data_label[index] = self.data_label[0]
            del index
            del img_name
            del img
            del img_path
        Batch_Norm = Batch_Normalization(self.data_images)
        label_one_hot = one_hot(self.data_label, _init_.classes_numbers)
        return Batch_Norm, label_one_hot

# # test
# data = get_images()
# img, label = data.get_mini_batch(train=False)
# plot_images(img, label, show_color="cool")

"""
import numpy as np
from DeepLearning.deep_learning import one_hot
from DeepLearning.Image import plot_images
import _init_


class get_images:
    def __init__(self):
        self.data = np.load(_init_.data_path)
        self.batch_size = _init_.batch_size
        self.count = 0
        self.length = self.data.shape[0]
        self.weight = _init_.input_image[0]
        self.height = _init_.input_image[1]
        self.data_images = np.zeros((self.batch_size, self.weight, self.height, 3))
        self.class_num = _init_.classes_numbers
        self.data_label = np.zeros(self.batch_size)

    def get_mini_batch(self):
        if (self.count+1)*self.batch_size < self.length:
            images, label = self.call_data()
            self.count += 1
            return images, label
        else:
            self.count = 0
            return self.call_data()

    def call_data(self):
        for kk in range(self.batch_size):
            self.data_images[kk, :, :, :] = np.reshape(
                self.data[self.count * self.batch_size + kk, :self.weight * self.height * 3],
                (self.height, self.weight, 3))
            self.data_label[kk] = self.data[self.count * self.batch_size + kk, self.weight * self.height * 3:]
        mean = np.mean(self.data_images)
        std = np.std(self.data_images)
        image_norm = (self.data_images - mean) / std
        label_one_hot = one_hot(self.data_label, self.class_num)
        return image_norm, label_one_hot

# data = get_images()
# for ii in range(5):
#     img, label = data.get_mini_batch()
#     plot_images(img, label, show_color="cool")

"""