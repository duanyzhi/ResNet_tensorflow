import tensorflow as tf
import _init_
from DeepLearning.deep_tensorflow import *


def Residual_Block(img, pre_img, weight, biases, offset, scale, strides=1):
    conv_img = tf.nn.conv2d(img, weight, strides=[1, strides, strides, 1], padding='SAME') + biases  # 3D [w, h, dim]
    input_shape = pre_img.get_shape()[3]    # [?, , , 64]
    output_shape = conv_img.get_shape()[3]  # [?, , , 256]
    if input_shape != output_shape:
        weight_pre = tf.get_variable('weight_pre', [1, 1, input_shape, output_shape], initializer=tf.random_normal_initializer(mean=0, stddev=1))
        biases_pre = tf.get_variable('biases_pre', [output_shape])
        conv_pre_img = tf.nn.conv2d(pre_img, weight_pre, strides=[1, strides, strides, 1], padding='SAME') + biases_pre
        output = conv_img + conv_pre_img
        _init_.parameters += [weight_pre, biases_pre]
    else:
        output = conv_img + pre_img
    mean, variance = tf.nn.moments(output, [0, 1, 2])
    conv_batch = tf.nn.batch_normalization(output, mean, variance, offset, scale, 1e-10)
    return tf.nn.relu(conv_batch)


def train_loss(prediction, labels):
    prediction = tf.nn.softmax(prediction)
    # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels)
    cross_entropy = -tf.reduce_sum(labels * tf.log(prediction))        # 求和
    # cross_entropy = tf.reduce_sum(prediction - labels)
    train_step = tf.train.GradientDescentOptimizer(_init_.learning_rate).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    return train_step, accuracy

"""
  in: [84, 84, 3]                           [32, 32, 3]
  conv1 :[21, 21, 64]                       [16, 16, 64]
  conv2: [10, 10, 256]
  conv3: [5, 5, 512]
  conv4: [2, 2, 1024]
  conv5: [1, 1, 2048]
  fc: []
"""


class ResNet:
    def __init__(self):
        self.img = None
        self.reuse = False
        self.learning_rate = _init_.learning_rate

    def __call__(self, img, scope):
        self.img = img         # [224, 224, 3]
        with tf.variable_scope(scope, reuse=self.reuse) as scope_name:
            if self.reuse:
                scope_name.reuse_variables()
            # conv1
            with tf.variable_scope('conv1'):
                weight_1 = tf.get_variable('weight', shape=[3, 3, 3, 64], initializer=tf.random_normal_initializer(mean=0, stddev=1))   # [k_size, k_size, input_size, output_size]
                biases_1 = tf.get_variable('biases', [64])
                offset_1 = tf.get_variable('offset', [64], initializer=tf.constant_initializer(0.0))
                scale_1 = tf.get_variable('scale', [64], initializer=tf.constant_initializer(1.0))
                conv1_ReLu = conv(self.img, weight_1, biases_1, offset_1, scale_1, strides=1)
                # conv1 = conv1_ReLu
                conv1 = max_pool(conv1_ReLu, k_size=(2, 2), stride=(2, 2))   # out [56, 56, 64]
                _init_.parameters += [weight_1, biases_1, offset_1, scale_1]
            # conv2
            with tf.variable_scope('conv2'):
                with tf.variable_scope('conv2_1'):
                    with tf.variable_scope('conv2_1_1'):
                        weight2_1_1 = tf.get_variable('weight', shape=[1, 1, 64, 64], initializer=tf.random_normal_initializer(mean=0, stddev=1))
                        biases2_1_1 = tf.get_variable('biases', [64])
                        offset2_1_1 = tf.get_variable('offset', [64], initializer=tf.constant_initializer(0.0))
                        scale2_1_1 = tf.get_variable('scale', [64], initializer=tf.constant_initializer(1.0))
                        conv2_1_1_ReLu = conv(conv1, weight2_1_1, biases2_1_1, offset2_1_1, scale2_1_1, strides=1)
                        _init_.parameters += [weight2_1_1, biases2_1_1, offset2_1_1, scale2_1_1]
                    with tf.variable_scope('conv2_1_2'):
                        weight2_1_2 = tf.get_variable('weight', [3, 3, 64, 64], initializer=tf.random_normal_initializer(mean=0, stddev=1))
                        biases2_1_2 = tf.get_variable('biases', [64])
                        offset2_1_2 = tf.get_variable('offset', [64], initializer=tf.constant_initializer(0.0))
                        scale2_1_2 = tf.get_variable('scale', [64], initializer=tf.constant_initializer(1.0))
                        conv2_1_2_ReLu = conv(conv2_1_1_ReLu, weight2_1_2, biases2_1_2, offset2_1_2, scale2_1_2, strides=1)
                        _init_.parameters += [weight2_1_2, biases2_1_2, offset2_1_2, scale2_1_2]
                    with tf.variable_scope('conv2_1_3'):
                        weight2_1_3 = tf.get_variable('weight', [1, 1, 64, 256], initializer=tf.random_normal_initializer(mean=0, stddev=1))
                        biases2_1_3 = tf.get_variable('biases', [256])
                        offset2_1_3 = tf.get_variable('offset', [256], initializer=tf.constant_initializer(0.0))
                        scale2_1_3 = tf.get_variable('scale', [256], initializer=tf.constant_initializer(1.0))
                        conv2_1 = Residual_Block(conv2_1_2_ReLu, conv1,  weight2_1_3, biases2_1_3, offset2_1_3,
                                                        scale2_1_3,  strides=1)
                        _init_.parameters += [weight2_1_3, biases2_1_3, offset2_1_3, scale2_1_3]
                with tf.variable_scope('conv2_2'):
                    with tf.variable_scope('conv2_2_1'):
                        weight2_2_1 = tf.get_variable('weight', [1, 1, 256, 64], initializer=tf.random_normal_initializer(mean=0, stddev=1))
                        biases2_2_1 = tf.get_variable('biases', [64])
                        offset2_2_1 = tf.get_variable('offset', [64], initializer=tf.constant_initializer(0.0))
                        scale2_2_1 = tf.get_variable('scale', [64], initializer=tf.constant_initializer(1.0))
                        conv2_2_1_ReLu = conv(conv2_1, weight2_2_1, biases2_2_1, offset2_2_1, scale2_2_1, strides=1)
                        _init_.parameters += [weight2_2_1, biases2_2_1, offset2_2_1, scale2_2_1]
                    with tf.variable_scope('conv2_2_2'):
                        weight2_2_2 = tf.get_variable('weight', [3, 3, 64, 64], initializer=tf.random_normal_initializer(mean=0, stddev=1))
                        biases2_2_2 = tf.get_variable('biases', [64])
                        offset2_2_2 = tf.get_variable('offset', [64], initializer=tf.constant_initializer(0.0))
                        scale2_2_2 = tf.get_variable('scale', [64], initializer=tf.constant_initializer(1.0))
                        conv2_2_2_ReLu = conv(conv2_2_1_ReLu, weight2_2_2, biases2_2_2, offset2_2_2, scale2_2_2, strides=1)
                        _init_.parameters += [weight2_2_2, biases2_2_2, offset2_2_2, scale2_2_2]
                    with tf.variable_scope('conv2_2_3'):
                        weight2_2_3 = tf.get_variable('weight', [1, 1, 64, 256], initializer=tf.random_normal_initializer(mean=0, stddev=1))
                        biases2_2_3 = tf.get_variable('biases', [256])
                        offset2_2_3 = tf.get_variable('offset', [256], initializer=tf.constant_initializer(0.0))
                        scale2_2_3 = tf.get_variable('scale', [256], initializer=tf.constant_initializer(1.0))
                        conv2_2 = Residual_Block(conv2_2_2_ReLu, conv2_1,  weight2_2_3, biases2_2_3, offset2_2_3,
                                                        scale2_2_3,  strides=1)
                        _init_.parameters += [weight2_2_3, biases2_2_3, offset2_2_3, scale2_2_3]
                with tf.variable_scope('conv2_3'):
                    with tf.variable_scope('conv2_3_1'):
                        weight2_3_1 = tf.get_variable('weight', [1, 1, 256, 64], initializer=tf.random_normal_initializer(mean=0, stddev=1))
                        biases2_3_1 = tf.get_variable('biases', [64])
                        offset2_3_1 = tf.get_variable('offset', [64], initializer=tf.constant_initializer(0.0))
                        scale2_3_1 = tf.get_variable('scale', [64], initializer=tf.constant_initializer(1.0))
                        conv2_3_1_ReLu = conv(conv2_2, weight2_3_1, biases2_3_1, offset2_3_1, scale2_3_1, strides=1)
                        _init_.parameters += [weight2_3_1, biases2_3_1, offset2_3_1, scale2_3_1]
                    with tf.variable_scope('conv2_3_2'):
                        weight2_3_2 = tf.get_variable('weight', [3, 3, 64, 64], initializer=tf.random_normal_initializer(mean=0, stddev=1))
                        biases2_3_2 = tf.get_variable('biases', [64])
                        offset2_3_2 = tf.get_variable('offset', [64], initializer=tf.constant_initializer(0.0))
                        scale2_3_2 = tf.get_variable('scale', [64], initializer=tf.constant_initializer(1.0))
                        conv2_3_2_ReLu = conv(conv2_3_1_ReLu, weight2_3_2, biases2_3_2, offset2_3_2, scale2_3_2, strides=1)
                        _init_.parameters += [weight2_3_2, biases2_3_2, offset2_3_2, scale2_3_2]
                    with tf.variable_scope('conv2_3_3'):
                        weight2_3_3 = tf.get_variable('weight', [1, 1, 64, 256], initializer=tf.random_normal_initializer(mean=0, stddev=1))
                        biases2_3_3 = tf.get_variable('biases', [256])
                        offset2_3_3 = tf.get_variable('offset', [256], initializer=tf.constant_initializer(0.0))
                        scale2_3_3 = tf.get_variable('scale', [256], initializer=tf.constant_initializer(1.0))
                        conv2_3 = Residual_Block(conv2_3_2_ReLu, conv2_2,  weight2_3_3, biases2_3_3, offset2_3_3,
                                                        scale2_3_3,  strides=1)
                        _init_.parameters += [weight2_3_3, biases2_3_3, offset2_3_3, scale2_3_3]
                        conv2 = max_pool(conv2_3, k_size=(2, 2), stride=(2, 2))    # [28, 28, 256]
            # conv3
            with tf.variable_scope('conv3'):
                with tf.variable_scope('conv3_1'):
                    with tf.variable_scope('conv3_1_1'):
                        weight3_1_1 = tf.get_variable('weight', [1, 1, 256, 128], initializer=tf.random_normal_initializer(mean=0, stddev=1))
                        biases3_1_1 = tf.get_variable('biases', [128])
                        offset3_1_1 = tf.get_variable('offset', [128], initializer=tf.constant_initializer(0.0))
                        scale3_1_1 = tf.get_variable('scale', [128], initializer=tf.constant_initializer(1.0))
                        conv3_1_1_ReLu = conv(conv2, weight3_1_1, biases3_1_1, offset3_1_1, scale3_1_1, strides=1)
                        _init_.parameters += [weight3_1_1, biases3_1_1, offset3_1_1, scale3_1_1]
                    with tf.variable_scope('conv3_1_2'):
                        weight3_1_2 = tf.get_variable('weight', [3, 3, 128, 128], initializer=tf.random_normal_initializer(mean=0, stddev=1))
                        biases3_1_2 = tf.get_variable('biases', [128])
                        offset3_1_2 = tf.get_variable('offset', [128], initializer=tf.constant_initializer(0.0))
                        scale3_1_2 = tf.get_variable('scale', [128], initializer=tf.constant_initializer(1.0))
                        conv3_1_2_ReLu = conv(conv3_1_1_ReLu, weight3_1_2, biases3_1_2, offset3_1_2, scale3_1_2, strides=1)
                        _init_.parameters += [weight3_1_2, biases3_1_2, offset3_1_2, scale3_1_2]
                    with tf.variable_scope('conv3_1_3'):
                        weight3_1_3 = tf.get_variable('weight', [1, 1, 128, 512], initializer=tf.random_normal_initializer(mean=0, stddev=1))
                        biases3_1_3 = tf.get_variable('biases', [512])
                        offset3_1_3 = tf.get_variable('offset', [512], initializer=tf.constant_initializer(0.0))
                        scale3_1_3 = tf.get_variable('scale', [512], initializer=tf.constant_initializer(1.0))
                        conv3_1 = Residual_Block(conv3_1_2_ReLu, conv2,  weight3_1_3, biases3_1_3, offset3_1_3,
                                                        scale3_1_3,  strides=1)
                        _init_.parameters += [weight3_1_3, biases3_1_3, offset3_1_3, scale3_1_3]
                with tf.variable_scope('conv3_2'):
                    with tf.variable_scope('conv3_2_1'):
                        weight3_2_1 = tf.get_variable('weight', [1, 1, 512, 128], initializer=tf.random_normal_initializer(mean=0, stddev=1))
                        biases3_2_1 = tf.get_variable('biases', [128])
                        offset3_2_1 = tf.get_variable('offset', [128], initializer=tf.constant_initializer(0.0))
                        scale3_2_1 = tf.get_variable('scale', [128], initializer=tf.constant_initializer(1.0))
                        conv3_2_1_ReLu = conv(conv3_1, weight3_2_1, biases3_2_1, offset3_2_1, scale3_2_1, strides=1)
                        _init_.parameters += [weight3_2_1, biases3_2_1, offset3_2_1, scale3_2_1]
                    with tf.variable_scope('conv3_2_2'):
                        weight3_2_2 = tf.get_variable('weight', [3, 3, 128, 128], initializer=tf.random_normal_initializer(mean=0, stddev=1))
                        biases3_2_2 = tf.get_variable('biases', [128])
                        offset3_2_2 = tf.get_variable('offset', [128], initializer=tf.constant_initializer(0.0))
                        scale3_2_2 = tf.get_variable('scale', [128], initializer=tf.constant_initializer(1.0))
                        conv3_2_2_ReLu = conv(conv3_2_1_ReLu, weight3_2_2, biases3_2_2, offset3_2_2, scale3_2_2, strides=1)
                        _init_.parameters += [weight3_2_2, biases3_2_2, offset3_2_2, scale3_2_2]
                    with tf.variable_scope('conv3_2_3'):
                        weight3_2_3 = tf.get_variable('weight', [1, 1, 128, 512], initializer=tf.random_normal_initializer(mean=0, stddev=1))
                        biases3_2_3 = tf.get_variable('biases', [512])
                        offset3_2_3 = tf.get_variable('offset', [512], initializer=tf.constant_initializer(0.0))
                        scale3_2_3 = tf.get_variable('scale', [512], initializer=tf.constant_initializer(1.0))
                        conv3_2 = Residual_Block(conv3_2_2_ReLu, conv3_1,  weight3_2_3, biases3_2_3, offset3_2_3,
                                                        scale3_2_3,  strides=1)
                        _init_.parameters += [weight3_2_3, biases3_2_3, offset3_2_3, scale3_2_3]
                with tf.variable_scope('conv3_3'):
                    with tf.variable_scope('conv3_3_1'):
                        weight3_3_1 = tf.get_variable('weight', [1, 1, 512, 128], initializer=tf.random_normal_initializer(mean=0, stddev=1))
                        biases3_3_1 = tf.get_variable('biases', [128])
                        offset3_3_1 = tf.get_variable('offset', [128], initializer=tf.constant_initializer(0.0))
                        scale3_3_1 = tf.get_variable('scale', [128], initializer=tf.constant_initializer(1.0))
                        conv3_3_1_ReLu = conv(conv3_2, weight3_3_1, biases3_3_1, offset3_3_1, scale3_3_1, strides=1)
                        _init_.parameters += [weight3_3_1, biases3_3_1, offset3_3_1, scale3_3_1]
                    with tf.variable_scope('conv3_3_2'):
                        weight3_3_2 = tf.get_variable('weight', [3, 3, 128, 128], initializer=tf.random_normal_initializer(mean=0, stddev=1))
                        biases3_3_2 = tf.get_variable('biases', [128])
                        offset3_3_2 = tf.get_variable('offset', [128], initializer=tf.constant_initializer(0.0))
                        scale3_3_2 = tf.get_variable('scale', [128], initializer=tf.constant_initializer(1.0))
                        conv3_3_2_ReLu = conv(conv3_3_1_ReLu, weight3_3_2, biases3_3_2, offset3_3_2, scale3_3_2, strides=1)
                        _init_.parameters += [weight3_3_2, biases3_3_2, offset3_3_2, scale3_3_2]
                    with tf.variable_scope('conv3_3_3'):
                        weight3_3_3 = tf.get_variable('weight', [1, 1, 128, 512], initializer=tf.random_normal_initializer(mean=0, stddev=1))
                        biases3_3_3 = tf.get_variable('biases', [512])
                        offset3_3_3 = tf.get_variable('offset', [512], initializer=tf.constant_initializer(0.0))
                        scale3_3_3 = tf.get_variable('scale', [512], initializer=tf.constant_initializer(1.0))
                        conv3_3 = Residual_Block(conv3_3_2_ReLu, conv3_2,  weight3_3_3, biases3_3_3, offset3_3_3,
                                                        scale3_3_3,  strides=1)
                        _init_.parameters += [weight3_3_3, biases3_3_3, offset3_3_3, scale3_3_3]
                with tf.variable_scope('conv3_4'):
                    with tf.variable_scope('conv3_4_1'):
                        weight3_4_1 = tf.get_variable('weight', [1, 1, 512, 128], initializer=tf.random_normal_initializer(mean=0, stddev=1))
                        biases3_4_1 = tf.get_variable('biases', [128])
                        offset3_4_1 = tf.get_variable('offset', [128], initializer=tf.constant_initializer(0.0))
                        scale3_4_1 = tf.get_variable('scale', [128], initializer=tf.constant_initializer(1.0))
                        conv3_4_1_ReLu = conv(conv3_3, weight3_4_1, biases3_4_1, offset3_4_1, scale3_4_1,
                                              strides=1)
                        _init_.parameters += [weight3_4_1, biases3_4_1, offset3_4_1, scale3_4_1]
                    with tf.variable_scope('conv3_4_2'):
                        weight3_4_2 = tf.get_variable('weight', [3, 3, 128, 128], initializer=tf.random_normal_initializer(mean=0, stddev=1))
                        biases3_4_2 = tf.get_variable('biases', [128])
                        offset3_4_2 = tf.get_variable('offset', [128], initializer=tf.constant_initializer(0.0))
                        scale3_4_2 = tf.get_variable('scale', [128], initializer=tf.constant_initializer(1.0))
                        conv3_4_2_ReLu = conv(conv3_4_1_ReLu, weight3_4_2, biases3_4_2, offset3_4_2, scale3_4_2,
                                              strides=1)
                        _init_.parameters += [weight3_4_2, biases3_4_2, offset3_4_2, scale3_4_2]
                    with tf.variable_scope('conv3_4_3'):
                        weight3_4_3 = tf.get_variable('weight', [1, 1, 128, 512], initializer=tf.random_normal_initializer(mean=0, stddev=1))
                        biases3_4_3 = tf.get_variable('biases', [512])
                        offset3_4_3 = tf.get_variable('offset', [512], initializer=tf.constant_initializer(0.0))
                        scale3_4_3 = tf.get_variable('scale', [512], initializer=tf.constant_initializer(1.0))
                        conv3_4 = Residual_Block(conv3_4_2_ReLu, conv3_3, weight3_4_3, biases3_4_3, offset3_4_3,
                                                 scale3_4_3, strides=1)
                        _init_.parameters += [weight3_4_3, biases3_4_3, offset3_4_3, scale3_4_3]

                        conv3 = max_pool(conv3_4, k_size=(2, 2), stride=(2, 2))    # [14, 14, 512]
            # conv4
            print(conv3)
            in_img = conv3
            with tf.variable_scope('conv4'):
                for kk in range(6):          # resnet50---6
                    with tf.variable_scope('conv4_' + str(kk)):
                        with tf.variable_scope('conv4_1'):
                            in_shape = in_img.get_shape()[3]
                            weight4_1 = tf.get_variable('weight', [1, 1, in_shape, 256])
                            biases4_1 = tf.get_variable('biases', [256])
                            offset4_1 = tf.get_variable('offset', [256], initializer=tf.constant_initializer(0.0))
                            scale4_1 = tf.get_variable('scale', [256], initializer=tf.constant_initializer(1.0))
                            conv4_1_ReLu = conv(in_img, weight4_1, biases4_1, offset4_1, scale4_1,
                                                  strides=1)
                            _init_.parameters += [weight4_1, biases4_1, offset4_1, scale4_1]
                        with tf.variable_scope('conv4_2'):
                            weight4_2 = tf.get_variable('weight', [3, 3, 256, 256])
                            biases4_2 = tf.get_variable('biases', [256])
                            offset4_2 = tf.get_variable('offset', [256], initializer=tf.constant_initializer(0.0))
                            scale4_2 = tf.get_variable('scale', [256], initializer=tf.constant_initializer(1.0))
                            conv4_2_ReLu = conv(conv4_1_ReLu, weight4_2, biases4_2, offset4_2, scale4_2,
                                                  strides=1)
                            _init_.parameters += [weight4_2, biases4_2, offset4_2, scale4_2]
                        with tf.variable_scope('conv4_3'):
                            weight4_3 = tf.get_variable('weight', [1, 1, 256, 1024])
                            biases4_3 = tf.get_variable('biases', [1024])
                            offset4_3 = tf.get_variable('offset', [1024], initializer=tf.constant_initializer(0.0))
                            scale4_3 = tf.get_variable('scale', [1024], initializer=tf.constant_initializer(1.0))
                            conv3_4 = Residual_Block(conv4_2_ReLu, in_img, weight4_3, biases4_3, offset4_3,
                                                     scale4_3, strides=1)
                            _init_.parameters += [weight4_3, biases4_3, offset4_3, scale4_3]
                    in_img = conv3_4
                conv4 = max_pool(in_img, k_size=(2, 2), stride=(2, 2))    # [7, 7, 1024]
            # conv5
            print(conv4)
            with tf.variable_scope('conv5'):
                with tf.variable_scope('conv5_1'):
                    with tf.variable_scope('conv5_1_1'):
                        weight5_1_1 = tf.get_variable('weight', [1, 1, 1024, 512])
                        biases5_1_1 = tf.get_variable('biases', [512])
                        offset5_1_1 = tf.get_variable('offset', [512], initializer=tf.constant_initializer(0.0))
                        scale5_1_1 = tf.get_variable('scale', [512], initializer=tf.constant_initializer(1.0))
                        conv5_1_1_ReLu = conv(conv4, weight5_1_1, biases5_1_1, offset5_1_1, scale5_1_1, strides=1)
                        _init_.parameters += [weight5_1_1, biases5_1_1, offset5_1_1, scale5_1_1]
                    with tf.variable_scope('conv5_1_2'):
                        weight5_1_2 = tf.get_variable('weight', [3, 3, 512, 512])
                        biases5_1_2 = tf.get_variable('biases', [512])
                        offset5_1_2 = tf.get_variable('offset', [512], initializer=tf.constant_initializer(0.0))
                        scale5_1_2 = tf.get_variable('scale', [512], initializer=tf.constant_initializer(1.0))
                        conv5_1_2_ReLu = conv(conv5_1_1_ReLu, weight5_1_2, biases5_1_2, offset5_1_2, scale5_1_2,
                                              strides=1)
                        _init_.parameters += [weight5_1_2, biases5_1_2, offset5_1_2, scale5_1_2]
                    with tf.variable_scope('conv5_1_3'):
                        weight5_1_3 = tf.get_variable('weight', [1, 1, 512, 2048])
                        biases5_1_3 = tf.get_variable('biases', [2048])
                        offset5_1_3 = tf.get_variable('offset', [2048], initializer=tf.constant_initializer(0.0))
                        scale5_1_3 = tf.get_variable('scale', [2048], initializer=tf.constant_initializer(1.0))
                        conv5_1 = Residual_Block(conv5_1_2_ReLu, conv4, weight5_1_3, biases5_1_3, offset5_1_3,
                                                 scale5_1_3, strides=1)
                        _init_.parameters += [weight5_1_3, biases5_1_3, offset5_1_3, scale5_1_3]
                with tf.variable_scope('conv5_2'):
                    with tf.variable_scope('conv5_2_1'):
                        weight5_2_1 = tf.get_variable('weight', [1, 1, 2048, 512])
                        biases5_2_1 = tf.get_variable('biases', [512])
                        offset5_2_1 = tf.get_variable('offset', [512], initializer=tf.constant_initializer(0.0))
                        scale5_2_1 = tf.get_variable('scale', [512], initializer=tf.constant_initializer(1.0))
                        conv5_2_1_ReLu = conv(conv5_1, weight5_2_1, biases5_2_1, offset5_2_1, scale5_2_1, strides=1)
                        _init_.parameters += [weight5_2_1, biases5_2_1, offset5_2_1, scale5_2_1]
                    with tf.variable_scope('conv5_2_2'):
                        weight5_2_2 = tf.get_variable('weight', [3, 3, 512, 512])
                        biases5_2_2 = tf.get_variable('biases', [512])
                        offset5_2_2 = tf.get_variable('offset', [512], initializer=tf.constant_initializer(0.0))
                        scale5_2_2 = tf.get_variable('scale', [512], initializer=tf.constant_initializer(1.0))
                        conv5_2_2_ReLu = conv(conv5_2_1_ReLu, weight5_2_2, biases5_2_2, offset5_2_2, scale5_2_2,
                                              strides=1)
                        _init_.parameters += [weight5_2_2, biases5_2_2, offset5_2_2, scale5_2_2]
                    with tf.variable_scope('conv5_2_3'):
                        weight5_2_3 = tf.get_variable('weight', [1, 1, 512, 2048])
                        biases5_2_3 = tf.get_variable('biases', [2048])
                        offset5_2_3 = tf.get_variable('offset', [2048], initializer=tf.constant_initializer(0.0))
                        scale5_2_3 = tf.get_variable('scale', [2048], initializer=tf.constant_initializer(1.0))
                        conv5_2 = Residual_Block(conv5_2_2_ReLu, conv5_1, weight5_2_3, biases5_2_3, offset5_2_3,
                                                 scale5_2_3, strides=1)
                        _init_.parameters += [weight5_2_3, biases5_2_3, offset5_2_3, scale5_2_3]
                with tf.variable_scope('conv5_3'):
                    with tf.variable_scope('conv5_3_1'):
                        weight5_3_1 = tf.get_variable('weight', [1, 1, 2048, 512])
                        biases5_3_1 = tf.get_variable('biases', [512])
                        offset5_3_1 = tf.get_variable('offset', [512], initializer=tf.constant_initializer(0.0))
                        scale5_3_1 = tf.get_variable('scale', [512], initializer=tf.constant_initializer(1.0))
                        conv5_3_1_ReLu = conv(conv5_2, weight5_3_1, biases5_3_1, offset5_3_1, scale5_3_1, strides=1)
                        _init_.parameters += [weight5_3_1, biases5_3_1, offset5_3_1, scale5_3_1]
                    with tf.variable_scope('conv5_3_2'):
                        weight5_3_2 = tf.get_variable('weight', [3, 3, 512, 512])
                        biases5_3_2 = tf.get_variable('biases', [512])
                        offset5_3_2 = tf.get_variable('offset', [512], initializer=tf.constant_initializer(0.0))
                        scale5_3_2 = tf.get_variable('scale', [512], initializer=tf.constant_initializer(1.0))
                        conv5_3_2_ReLu = conv(conv5_3_1_ReLu, weight5_3_2, biases5_3_2, offset5_3_2, scale5_3_2,
                                              strides=1)
                        _init_.parameters += [weight5_3_2, biases5_3_2, offset5_3_2, scale5_3_2]
                    with tf.variable_scope('conv2_3_3'):
                        weight5_3_3 = tf.get_variable('weight', [1, 1, 512, 2048])
                        biases5_3_3 = tf.get_variable('biases', [2048])
                        offset5_3_3 = tf.get_variable('offset', [2048], initializer=tf.constant_initializer(0.0))
                        scale5_3_3 = tf.get_variable('scale', [2048], initializer=tf.constant_initializer(1.0))
                        conv5_3 = Residual_Block(conv5_3_2_ReLu, conv5_2, weight5_3_3, biases5_3_3, offset5_3_3,
                                                 scale5_3_3, strides=1)
                        _init_.parameters += [weight5_3_3, biases5_3_3, offset5_3_3, scale5_3_3]
                        conv5 = max_pool(conv5_3, k_size=(2, 2), stride=(2, 2))  # [7, 7, 2048]
            # conv5
<<<<<<< HEAD
            print(conv5)
            # average_pool = ave_pool(conv5, k_size=(7, 7))
            # print(average_pool)
            ave_pooling = tf.squeeze(conv5, [1, 2])
=======
            ave_pool = tf.squeeze(conv5, [1, 2])
            # ave_pool = tf.clip_by_value(ave_pool, -1, 1)
            # k = conv5.get_shape()[1]
            # ave_pool = max_pool(conv5, k_size=(int(k), int(k)), stride=(1, 1))  # [7, 7, 2048][1, 1, 2048]
            # ave_pool = tf.squeeze(ave_pool, [1, 2])  # in [batch_size, 1, 1, 2048]  out: [batch_size, 2048]
>>>>>>> d4a73fb6e399cca8b0646e8dd37f790d5888d3f0
            with tf.variable_scope('fc'):
                weight_fc2 = tf.get_variable('weight', [2048, _init_.classes_numbers])
                biases_fc2 = tf.get_variable('biases', [_init_.classes_numbers])
                fc = tf.nn.bias_add(tf.matmul(ave_pooling, weight_fc2), biases_fc2)
                mean, variance = tf.nn.moments(fc, [0, 1])
                fc = tf.nn.batch_normalization(fc, mean, variance, 0, 1, 1e-10)
                print(fc)
                _init_.parameters += [weight_fc2, biases_fc2]
        self.reuse = True
        return fc





