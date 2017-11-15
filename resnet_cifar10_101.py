from __future__ import division
from DeepLearning.deep_tensorflow import *
from _init_ import FLAGS


def _decay():
    """L2 weight decay loss."""
    costs = []
    for var in tf.trainable_variables():
        print("------------", var)
        costs.append(tf.nn.l2_loss(var))
        # tf.summary.histogram(var.op.name, var)
    return tf.multiply(FLAGS.weight_decay, tf.add_n(costs))


def train_loss(prediction, labels, optimizer="Mom"):
    # demo1
    # prediction = tf.nn.softmax(prediction)
    # cross_entropy = -tf.reduce_sum(labels * tf.log(prediction))        # 求和
    # demo2
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels)
    cross_entropy_loss = tf.reduce_sum(cross_entropy)
    weight_decay_loss = _decay()
    loss = cross_entropy_loss + weight_decay_loss
    if optimizer == "Mom":  # Momentum
        train_step = tf.train.MomentumOptimizer(FLAGS.learning_rate, 0.9).minimize(loss)
    elif optimizer == 'RMSProp':
        train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(loss)
    else:        # SGD
        train_step = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(loss)

    # ----------------------------------------------------------------------------
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    tf.summary.scalar("Loss_Function", loss)
    tf.summary.scalar("acc", accuracy)
    return train_step, accuracy


def Residual_block(in_img, filters, down_sample, projection=False):
    input_depth = int(in_img.get_shape()[3])
    if down_sample:
        stride = 2
    else:
        stride = 1
    # if down_sample:
    #     filter_ = [1, 2, 2, 1]
    #     in_img = tf.nn.max_pool(in_img, ksize=filter_, strides=filter_, padding='SAME')

    with tf.variable_scope('conv1'):
        weight_1 = create_variables(name='weight_1', shape=[3, 3, input_depth, filters])
        biases_1 = create_variables(name='biases_1', shape=[filters], initializer=tf.zeros_initializer())
        conv1_ReLu = Conv_BN_Relu(in_img, weight_1, biases_1, filters, strides=stride)

    with tf.variable_scope('conv2'):
        weight_2 = create_variables(name='weight_2', shape=[3, 3, filters, filters])
        biases_2 = create_variables(name='biases_2', shape=[filters], initializer=tf.zeros_initializer())
        conv2_ReLu = Conv_BN_Relu(conv1_ReLu, weight_2, biases_2, filters, strides=1)

    if input_depth != filters:
        if projection:
            # Option B: Projection shortcut
            weight_3 = create_variables(name='weight_3', shape=[1, 1, input_depth, filters])
            biases_3 = create_variables(name='biases_3', shape=[filters], initializer=tf.zeros_initializer())
            input_layer = Conv_BN_Relu(in_img, weight_3, biases_3, filters, strides=stride)
        else:
            # Option A: Zero-padding
            if down_sample:
                in_img = ave_pool(in_img)
            input_layer = tf.pad(in_img, [[0, 0], [0, 0], [0, 0], [int((filters - input_depth)/2), filters - input_depth - int((filters - input_depth)/2)]])  # 维度是4维[batch_size, :, :, dim] 我么要pad dim的维度
    else:
        input_layer = in_img

    output = conv2_ReLu + input_layer
    output = tf.nn.relu(output)
    return output


def create_variables(name, shape, initializer=tf.contrib.layers.xavier_initializer(), is_fc_layer=False):
    '''
    :param name: A string. The name of the new variable
    :param shape: A list of dimensions
    :param initializer: User Xavier as default.
    :param is_fc_layer: Want to create fc layer variable? May use different weight_decay for fc
    layers.
    :return: The created variable
    '''

    # TODO: to allow different weight decay to fully connected layer and conv layer
    if is_fc_layer is True:
        regularizer = tf.contrib.layers.l2_regularizer(scale=FLAGS.weight_decay)
    else:
        regularizer = tf.contrib.layers.l2_regularizer(scale=FLAGS.weight_decay)

    new_variables = tf.get_variable(name, shape=shape, initializer=initializer,
                                    regularizer=regularizer, trainable=True)
    return new_variables




"""
  resnet-cifar10
  in: [32, 32, 3]
  conv: [32， 32， 16]
  conv1 :[32, 32, 16*k]
  conv2: [16, 16, 32*k]
  conv3: [8, 8, 64*k]
  ave-pool : [8*8] pooling----[1*1]
  fc: [64*k, 10]
"""


class ResNet:
    def __init__(self):
        self.img = None
        self.reuse = False
        self.learning_rate = FLAGS.learning_rate
        self.k = 1    # Original architecture
        self.filter = [16, 16*self.k, 32*self.k, 64*self.k]

    def __call__(self, img, scope):
        self.img = img         # [224, 224, 3]
        with tf.variable_scope(scope, reuse=self.reuse) as scope_name:
            if self.reuse:
                scope_name.reuse_variables()
            # conv1
            with tf.variable_scope('conv_pre'):   # 32
                weight_1 = create_variables(name='weight', shape=[3, 3, 3, self.filter[0]])
                biases_1 = create_variables(name='biases', shape=[self.filter[0]], initializer=tf.zeros_initializer())
                conv1_ReLu = tf.nn.conv2d(self.img, weight_1, strides=[1, 1, 1, 1], padding='SAME') + biases_1
                conv1_BN = Batch_Normalization(conv1_ReLu, self.filter[0])
                conv1 = tf.nn.relu(conv1_BN)
            # conv2
            in_img = conv1
            for ii in range(1, 4):
                with tf.variable_scope('conv' + str(ii)):   # 64
                    for kk in range(16):
                        down_sample = True if kk == 0 and ii != 1 else False
                        with tf.variable_scope('con_' + str(kk)):
                            in_img = Residual_block(in_img, filters=self.filter[ii], down_sample=down_sample)

            with tf.variable_scope('fc'):
                mean, variance = tf.nn.moments(in_img, axes=[0, 1, 2])
                beta = tf.get_variable('beta', self.filter[3], tf.float32,
                                       initializer=tf.constant_initializer(0.0, tf.float32))
                gamma = tf.get_variable('gamma', self.filter[3], tf.float32,
                                        initializer=tf.constant_initializer(1.0, tf.float32))
                bn_layer = tf.nn.batch_normalization(in_img, mean, variance, beta, gamma, 0.0001)
                relu_layer = tf.nn.relu(bn_layer)
                global_pool = tf.reduce_mean(relu_layer, [1, 2])
                # global_pool = ave_pool(relu_layer, k_size=(8, 8))
                print("global_pool", global_pool)

                weight_fc = tf.get_variable('fc_weight', [self.filter[3], FLAGS.classes_numbers], initializer=tf.uniform_unit_scaling_initializer(factor=1.0), regularizer=tf.contrib.layers.l2_regularizer(scale=FLAGS.weight_decay))
                biases_fc = tf.get_variable('fc_biases', [FLAGS.classes_numbers], initializer=tf.zeros_initializer(), regularizer=tf.contrib.layers.l2_regularizer(scale=FLAGS.weight_decay))
                fc = tf.matmul(global_pool, weight_fc) + biases_fc

                mean, variance = tf.nn.moments(fc, [0, 1])
                fc = tf.nn.batch_normalization(fc, mean, variance, 0, 1, 0.0001)
                print(fc)
        self.reuse = True
        return fc


# 0.89


