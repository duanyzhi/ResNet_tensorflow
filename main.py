# ------------------------------------------------------------------------ #
# ResNet101                                                                #
# Reference:                                                               #
# python 3.5                                                               #
# tensorflow1.1.0                                                          #
#                                                                          #
# 2017/8/29                                                                #
# Data: ImageNet                                                           #
# ------------------------------------------------------------------------ #
import tensorflow as tf

from get_images import get_images
from DeepLearning.deep_learning import learning_rate
from resnet import ResNet, train_loss

import _init_

cnn = ResNet()
data = get_images()
# tf.reset_default_graph()
x = tf.placeholder(tf.float32, [None, _init_.input_image[0], _init_.input_image[1], _init_.input_image[2]])
y = tf.placeholder(tf.float32, [None, _init_.classes_numbers])
fc_out = cnn(x, scope='resnet')
train_step, acc = train_loss(fc_out, y)

# saver = tf.train.Saver()
config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
# 开始不会给tensorflow全部gpu资源 而是按需增加
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    accs = []
    for kk in range(1, _init_.iteration_numbers):
        img_batch, label_batch = data.get_mini_batch()
        _init_.learning_rate = learning_rate(kk*_init_.batch_size, "epochs", _init_.learning_rate, _init_.epochs_number)
        _, accuracy = sess.run([train_step, acc], feed_dict={x: img_batch, y: label_batch})
        if kk % _init_.display_step == 0:
            accs.append(accuracy)
            print("Batch: ", kk, "Accuracy: ", accuracy)

