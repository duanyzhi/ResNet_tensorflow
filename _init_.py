# some init in here
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('weight_decay', 0.0001, '''scale for l2 regularization''')  # float类型
tf.app.flags.DEFINE_float('learning_rate', 0.01, ''' Initialization learning rate''')
tf.app.flags.DEFINE_integer('epochs_number', 50000, '''epochs number of image''')    # integer类型
tf.app.flags.DEFINE_integer('batch_size', 128, '''Batch Size''')
tf.app.flags.DEFINE_integer('iteration_numbers', 100000, '''Iteration Number of training''')
tf.app.flags.DEFINE_integer('display_step', 100, '''Display number to show the acc''')
tf.app.flags.DEFINE_float('input_image', [32, 32, 3], '''Input image size [weight, height, depth]''')
tf.app.flags.DEFINE_integer('classes_numbers', 10, '''classes number of model''')
tf.app.flags.DEFINE_string('log_dir', "tensorboard\\", '''Path to save log and checkpoint''')  # string类型



