import tensorflow as tf
import numpy as np
from DeepLearning.deep_learning import one_hot
from DeepLearning.Image import plot_images
import _init_
import matplotlib.pyplot as plt


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
