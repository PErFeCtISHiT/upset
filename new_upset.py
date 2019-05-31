import tensorflow as tf
import numpy as np
import keras
import loader


def get_non_trainable_variable(input_variable, session):
    return tf.Variable(session.run(input_variable), trainable=False)


sess = tf.Session()
image_width = 28
class_num = 10

fashion_mnist = keras.datasets.fashion_mnist
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
(train_images, train_labels), (test_images, test_labels) = loader.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0
train_labels = sess.run(tf.one_hot(train_labels, class_num))
test_labels = sess.run(tf.one_hot(test_labels, class_num))

target_train_x = tf.placeholder(tf.float32, shape=(None, image_width, image_width), name='target_train_x')
target_train_tag = tf.placeholder(tf.float32, shape=(None, class_num), name='target_train_tag')

flatten_num = image_width ** 2

flatten = tf.reshape(target_train_x, [-1, flatten_num])

w1_t = tf.Variable(tf.random_normal([784, 128], stddev=1, seed=1))
w2_t = tf.Variable(tf.random_normal([128, 10], stddev=1, seed=1))

output = target_train_x
output = tf.matmul(output, w1_t)
output = tf.nn.relu(output)
output = tf.matmul(output, w2_t)
output = tf.nn.softmax(output)

target_dataset_size = len(train_images)
target_batch_size = 8
target_steps = target_dataset_size // target_batch_size
target_steps = target_steps if target_dataset_size % target_batch_size == 0 else target_steps + 1

init_op = tf.global_variables_initializer()
sess.run(init_op)

