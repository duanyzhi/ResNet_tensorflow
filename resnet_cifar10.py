import tensorflow as tf
import _init_
from DeepLearning.deep_tensorflow import *
from _init_ import FLAGS

def train_loss(prediction, labels):
    # demo1
    # prediction = tf.nn.softmax(prediction)
    # cross_entropy = -tf.reduce_sum(labels * tf.log(prediction))        # 求和
    # demo2
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels)
    cross_entropy_loss = tf.reduce_sum(cross_entropy)
    regu_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

    loss = tf.add_n([cross_entropy_loss] + regu_losses)

    # SGD
    # train_step = tf.train.GradientDescentOptimizer(_init_.learning_rate).minimize(cross_entropy)
    # Momentum
    optimizer = tf.train.MomentumOptimizer(_init_.learning_rate, 0.9)
    train_step = optimizer.minimize(loss, global_step=_init_.global_step)
    # ----------------------------------------------------------
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    return train_step, accuracy


def create_variables(name, shape, initializer=tf.contrib.layers.xavier_initializer(), is_fc_layer=False):
    '''
    :param name: A string. The name of the new variable
    :param shape: A list of dimensions
    :param initializer: User Xavier as default.
    :param is_fc_layer: Want to create fc layer variable? May use different weight_decay for fc
    layers.
    :return: The created variable
    '''

    ## TODO: to allow different weight decay to fully connected layer and conv layer
    if is_fc_layer is True:
        regularizer = tf.contrib.layers.l2_regularizer(scale=FLAGS.weight_decay)
    else:
        regularizer = tf.contrib.layers.l2_regularizer(scale=FLAGS.weight_decay)

    new_variables = tf.get_variable(name, shape=shape, initializer=initializer,
                                    regularizer=regularizer)
    return new_variables

"""
  in: [32, 32, 3]
  conv1 :[21, 21, 64]
  conv2: [10, 10, 256]
  conv3: [5, 5, 512]

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
            with tf.variable_scope('conv_pre'):   # 1
                weight_1 = create_variables(name='weight', shape=[3, 3, 3, 16])
                biases_1 = tf.get_variable('biases', [16])
                conv1_ReLu = tf.nn.conv2d(self.img, weight_1, strides=[1, 1, 1, 1], padding='SAME') + biases_1
                conv1 = conv1_ReLu
                # conv1 = max_pool(conv1_ReLu, k_size=(2, 2), stride=(2, 2))   # out [32, 32, 16]
            print(conv1)
            # conv2
            in_img = conv1
            with tf.variable_scope('conv1'):   # 2  --  34
                for kk in range(16):         # resnet50---6
                    with tf.variable_scope('conv1_' + str(kk)):
                        with tf.variable_scope('conv1'):
                            in_shape = in_img.get_shape()[3]
                            weight_1 = tf.get_variable('weight', [3, 3, in_shape, 16], initializer=tf.contrib.layers.xavier_initializer())
                            biases_1 = tf.get_variable('biases', [16], initializer=tf.contrib.layers.xavier_initializer())
                            conv1_ReLu = BN_ReLU_Conv(in_img, weight_1, biases_1, in_shape, strides=1)
                        with tf.variable_scope('conv2'):
                            weight_2 = tf.get_variable('weight', [3, 3, 16, 16], initializer=tf.contrib.layers.xavier_initializer())
                            biases_2 = tf.get_variable('biases', [16], initializer=tf.contrib.layers.xavier_initializer())
                            conv2_ReLu = BN_ReLU_Conv(conv1_ReLu, weight_2, biases_2, 16, strides=1)
                        out_shape = conv2_ReLu.get_shape()[3]
                        if in_shape * 2 == out_shape:
                            pooled_input = tf.nn.avg_pool(in_img, ksize=[1, 2, 2, 1],
                                                          strides=[1, 2, 2, 1], padding='VALID')
                            # demo1
                            y = tf.zeros_like(pooled_input)
                            padded_input = tf.concat([pooled_input, y], axis=3)
                            # -----------------------------------------------------------------------------
                            # demo2
                            # weight_pad = tf.get_variable('weight', [1, 1, in_shape, out_shape],
                            #                            initializer=tf.contrib.layers.xavier_initializer())
                            # padded_input = tf.nn.conv2d(pooled_input, weight_pad, strides=[1, 1, 1, 1],
                            #                           padding='SAME')
                            # -----------------------------------------------------------------------------
                            print(padded_input)
                            # padded_input = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0], [in_shape // 2,
                            #                                                               in_shape // 2]])
                        else:
                            padded_input = in_img

                        output = conv2_ReLu + padded_input
                    in_img = output
                # in_img = max_pool(in_img)
                print(in_img)

            with tf.variable_scope('conv2'):  # 35 66       [32, 32, 16]
                for kk in range(16):
                    with tf.variable_scope('conv2_' + str(kk)):
                        with tf.variable_scope('conv1'):
                            in_shape = in_img.get_shape()[3]
                            if in_shape*2 == 32:
                                stride = 2
                            else:
                                stride = 1
                            weight_1 = tf.get_variable('weight', [3, 3, in_shape, 32],
                                                       initializer=tf.contrib.layers.xavier_initializer())
                            biases_1 = tf.get_variable('biases', [32], initializer=tf.contrib.layers.xavier_initializer())
                            conv1_ReLu = BN_ReLU_Conv(in_img, weight_1, biases_1, in_shape, strides=stride)
                        with tf.variable_scope('conv2'):
                            weight_2 = tf.get_variable('weight', [3, 3, 32, 32],
                                                       initializer=tf.contrib.layers.xavier_initializer())
                            biases_2 = tf.get_variable('biases', [32], initializer=tf.contrib.layers.xavier_initializer())
                            conv2_ReLu = BN_ReLU_Conv(conv1_ReLu, weight_2, biases_2, 32, strides=1)
                        out_shape = conv2_ReLu.get_shape()[3]
                        if in_shape * 2 == out_shape:
                            pooled_input = tf.nn.avg_pool(in_img, ksize=[1, 2, 2, 1],
                                                          strides=[1, 2, 2, 1], padding='VALID')
                            # demo1
                            y = tf.zeros_like(pooled_input)
                            padded_input = tf.concat([pooled_input, y], axis=3)
                            # -----------------------------------------------------------------------------
                            # demo2
                            # weight_pad = tf.get_variable('weight', [1, 1, in_shape, out_shape],
                            #                            initializer=tf.contrib.layers.xavier_initializer())
                            # padded_input = tf.nn.conv2d(pooled_input, weight_pad, strides=[1, 1, 1, 1],
                            #                           padding='SAME')
                            # -----------------------------------------------------------------------------
                            print(padded_input)
                        else:
                            padded_input = in_img

                        output = conv2_ReLu + padded_input
                    in_img = output
            print(in_img)
            with tf.variable_scope('conv3'):  # 35 66       [16, 16, 32]
                for kk in range(16):
                    with tf.variable_scope('conv3_' + str(kk)):
                        with tf.variable_scope('conv1'):
                            in_shape = in_img.get_shape()[3]
                            if in_shape * 2 == 64:
                                stride = 2
                            else:
                                stride = 1
                            weight_1 = tf.get_variable('weight', [3, 3, in_shape, 64],
                                                       initializer=tf.contrib.layers.xavier_initializer())
                            biases_1 = tf.get_variable('biases', [64], initializer=tf.contrib.layers.xavier_initializer())
                            conv1_ReLu = BN_ReLU_Conv(in_img, weight_1, biases_1, in_shape, strides=stride)
                        with tf.variable_scope('conv2'):
                            weight_2 = tf.get_variable('weight', [3, 3, 64, 64],
                                                       initializer=tf.contrib.layers.xavier_initializer())
                            biases_2 = tf.get_variable('biases', [64], initializer=tf.contrib.layers.xavier_initializer())
                            conv2_ReLu = BN_ReLU_Conv(conv1_ReLu, weight_2, biases_2, 64, strides=1)
                        out_shape = conv2_ReLu.get_shape()[3]
                        if in_shape * 2 == out_shape:
                            pooled_input = tf.nn.avg_pool(in_img, ksize=[1, 2, 2, 1],
                                                          strides=[1, 2, 2, 1], padding='VALID')
                            # demo1
                            y = tf.zeros_like(pooled_input)
                            padded_input = tf.concat([pooled_input, y], axis=3)
                            # -----------------------------------------------------------------------------
                            # demo2
                            # weight_pad = tf.get_variable('weight', [1, 1, in_shape, out_shape],
                            #                            initializer=tf.contrib.layers.xavier_initializer())
                            # padded_input = tf.nn.conv2d(pooled_input, weight_pad, strides=[1, 1, 1, 1],
                            #                           padding='SAME')
                            # -----------------------------------------------------------------------------
                            print(padded_input)
                        else:
                            padded_input = in_img

                        output = conv2_ReLu + padded_input
                    in_img = output     # [8, 8, 64]
            print(in_img)
            # out
            # average_pool = ave_pool(in_img, k_size=(8, 8))
            # print(average_pool)
            # ave_pooling = tf.squeeze(average_pool, [1, 2])
            with tf.variable_scope('fc'):
                mean, variance = tf.nn.moments(in_img, axes=[0, 1, 2])
                beta = tf.get_variable('beta', 64, tf.float32,
                                       initializer=tf.constant_initializer(0.0, tf.float32))
                gamma = tf.get_variable('gamma', 64, tf.float32,
                                        initializer=tf.constant_initializer(1.0, tf.float32))
                bn_layer = tf.nn.batch_normalization(in_img, mean, variance, beta, gamma, 1e-10)
                relu_layer = tf.nn.relu(bn_layer)
                global_pool = tf.reduce_mean(relu_layer, [1, 2])
                print("global_pool", global_pool)
                weight_fc2 = tf.get_variable('weight', [64, _init_.classes_numbers], initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
                biases_fc2 = tf.get_variable('biases', [_init_.classes_numbers], initializer=tf.zeros_initializer())
                fc = tf.nn.bias_add(tf.matmul(global_pool, weight_fc2), biases_fc2)

                mean, variance = tf.nn.moments(fc, [0, 1])
                fc = tf.nn.batch_normalization(fc, mean, variance, 0, 1, 1e-10)

                print(fc)
        self.reuse = True
        return fc


# 0.83


