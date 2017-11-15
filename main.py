# ------------------------------------------------------------------------ #
# ResNet101                                                                #
# Reference:                                                               #
# python 3.5                                                               #
# tensorflow1.1.0                                                          #
#                                                                          #
# 2017/8/29                                                                #
# Data: ImageNet                                                           #
# ------------------------------------------------------------------------ #
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import get_mnist
import math
import os

from resnet_cifar10_101 import ResNet, train_loss
from tensorflow.python.framework import ops
from DeepLearning.python import list_save
from python import _learning_rate_
from _init_ import FLAGS


ops.reset_default_graph()
cnn = ResNet()
data = get_mnist.cifar10()
x = tf.placeholder(tf.float32, [None, FLAGS.input_image[0], FLAGS.input_image[1], FLAGS.input_image[2]])
y = tf.placeholder(tf.float32, [None, FLAGS.classes_numbers])
tf.summary.image("input_image", x, 3)  # 图片写入tensorboard

fc_out = cnn(x, scope='resnet')
train_step, accuracy = train_loss(fc_out, y)

if not os.path.exists('tensorboard'):
    os.makedirs('tensorboard')
merged_summary_op = tf.summary.merge_all()

embedding = tf.Variable(tf.zeros([100, 10]), name="test_embedding")
assignment = embedding.assign(fc_out)

saver = tf.train.Saver()

config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
# 开始不会给tensorflow全部gpu资源 而是按需增加
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())

    summary_writer = tf.summary.FileWriter(FLAGS.log_dir + 'mnist_logs')
    summary_writer.add_graph(sess.graph)

    config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
    embedding_config = config.embeddings.add()
    embedding_config.tensor_name = embedding.name
    embedding_config.sprite.image_path = FLAGS.log_dir + 'sprite_1024.png'
    embedding_config.metadata_path = FLAGS.log_dir + 'labels_1024.tsv'
    # Specify the width and height of a single thumbnail.
    embedding_config.sprite.single_image_dim.extend([32, 32])
    tf.contrib.tensorboard.plugins.projector.visualize_embeddings(summary_writer, config)

    Acc_1 = []
    Acc_5 = []
    for kk in range(1, FLAGS.iteration_numbers):
        img_batch, label_batch = data.get_mini_batch(data_name="train")
        FLAGS.learning_rate = _learning_rate_(kk)
        _, acc_top_1, fc_, summary = sess.run([train_step, accuracy, fc_out, merged_summary_op], feed_dict={x: img_batch, y: label_batch})
        summary_writer.add_summary(summary, kk)

        if kk % FLAGS.display_step == 0:
            acc_1_ave = 0
            number_test = math.ceil(10000/FLAGS.batch_size)  # 向上取整
            for jj in range(number_test):
                img_batch, label_batch = data.get_mini_batch(data_name="test")
                acc_top_1 = sess.run(accuracy, feed_dict={x: img_batch, y: label_batch})
                acc_1_ave += acc_top_1
            saver.save(sess, FLAGS.log_dir + 'model.ckpt', global_step=10000)
            Acc_1.append(acc_1_ave/number_test)
            list_save(Acc_1, "data\\acc\\acc_CIFAR10_resnet100_26.txt")
            print("Batch: %-6d" % kk, "  ||  ", "Accuracy: %.6f" % (acc_1_ave / number_test), "  ||  ", "learning_rate", FLAGS.learning_rate)

    plt.plot(np.arange(len(Acc_1)), Acc_1, "g-", markersize=10)
    plt.xlabel('iter')
    plt.ylabel('Acc')
    plt.ylim([0.0, 1.0])
    plt.title('ResNet_100_Cifar10')
    plt.savefig("data\\fig\\ResNet_100_Cifar10_26.jpg")  # 保存图片
    plt.show()

# tensorboard --logdir=tensorboard/mnist_logs
# localhost:6006
# 0.91
