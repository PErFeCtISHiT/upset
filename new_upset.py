import keras
import numpy as np
import tensorflow as tf

import loader


def get_weight(shape):
    var = tf.Variable(tf.random_normal(shape), dtype=tf.float32, trainable=True)
    return var


def get_model(w1, w2, x):
    flatten_num = image_width ** 2
    layer = tf.reshape(x, [-1, flatten_num])
    layer = tf.matmul(layer, w1)
    layer = tf.nn.relu(layer)
    layer = tf.matmul(layer, w2)
    return layer


def print_num_of_total_parameters(output_detail=False, output_to_logging=False):
    total_parameters = 0
    parameters_string = ""

    for variable in tf.trainable_variables():

        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
        if len(shape) == 1:
            parameters_string += ("%s %d, " % (variable.name, variable_parameters))
        else:
            parameters_string += ("%s %s=%d, " % (variable.name, str(shape), variable_parameters))

    if output_to_logging:
        if output_detail:
            print(parameters_string)
        print("Total %d variables, %s params" % (len(tf.trainable_variables()), "{:,}".format(total_parameters)))
    else:
        if output_detail:
            print(parameters_string)
        print("Total %d variables, %s params" % (len(tf.trainable_variables()), "{:,}".format(total_parameters)))


# sess = tf_debug.LocalCLIDebugWrapperSession(sess)
image_width = 28
class_num = 10

fashion_mnist = keras.datasets.fashion_mnist
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
(train_images, train_labels), (test_images, test_labels) = loader.load_data()
train_images = (train_images / 255.0 - 0.5) * 2
test_images = (test_images / 255.0 - 0.5) * 2

dataset_size = len(train_images)

batch_size = 8
check_interval = 1000
steps = dataset_size // batch_size
steps = steps if dataset_size % batch_size == 0 else steps + 1

# t
train_x = tf.placeholder(tf.float32, shape=(None, 10), name='x-input')
# x
train_y = tf.placeholder(tf.float32, shape=(None, 28, 28), name='y-input')

arg_s = 1
arg_w = 0.06

# layer_dimension = [10, 128, 256, 512, 1024, 512, 784]
# layer1 = tf.Variable(tf.random_normal([10, 128], stddev=2))
# layer2 = tf.Variable(tf.random_normal([128, 256], stddev=2))
# layer3 = tf.Variable(tf.random_normal([256, 512], stddev=2))
# layer4 = tf.Variable(tf.random_normal([512, 1024], stddev=2))
# layer5 = tf.Variable(tf.random_normal([1024, 512], stddev=2))
# layer6 = tf.Variable(tf.random_normal([512, 784], stddev=2))
# num_layers = len(layer_dimension)
# current_layer = train_x
#
# current_layer = tf.nn.relu(tf.matmul(current_layer, layer1))
# current_layer = tf.nn.leaky_relu(tf.matmul(current_layer, layer2))
# current_layer = tf.nn.leaky_relu(tf.matmul(current_layer, layer3))
# current_layer = tf.nn.leaky_relu(tf.matmul(current_layer, layer4))
# current_layer = tf.nn.leaky_relu(tf.matmul(current_layer, layer5))
# current_layer = tf.tanh(tf.matmul(current_layer, layer6))
# current_layer = tf.reshape(current_layer, [-1, 28, 28])

# def fully_connected(prev_layer, num_units, is_training):
#     # batch_normalization
#     gamma = tf.Variable(tf.ones([num_units]))
#     beta = tf.Variable(tf.zeros([num_units]))
#     epsilon = 1e-3

current_layer = train_x
w1_upset = tf.Variable(tf.random_normal([10, 128], stddev=2))
bias1 = tf.Variable(tf.constant(0.1, shape=[128]))
w2_upset = tf.Variable(tf.random_normal([128, 256], stddev=2))
bias2 = tf.Variable(tf.constant(0.1, shape=[256]))
w3_upset = tf.Variable(tf.random_normal([256, 784], stddev=2))
bias3 = tf.Variable(tf.constant(0.1, shape=[784]))
current_layer = tf.nn.relu(tf.matmul(current_layer, w1_upset) + bias1)
current_layer = tf.nn.relu(tf.matmul(current_layer, w2_upset) + bias2)
current_layer = tf.tanh(tf.matmul(current_layer, w3_upset) + bias3)
current_layer = tf.reshape(current_layer, [-1, 28, 28])
output_layer = current_layer
new_image = tf.maximum(tf.minimum(arg_s * output_layer + train_y, 1), -1)

w1_n = tf.convert_to_tensor(np.load('w1.npy'))
w2_n = tf.convert_to_tensor(np.load('w2.npy'))

model = get_model(w1_n, w2_n, new_image)

# model = tf.nn.softmax(model)
# lc = -tf.reduce_mean(train_x * tf.log(tf.clip_by_value(model, 1e-10, 1.0)))

lc = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(labels=train_x, logits=model))
lf = arg_w * tf.reduce_mean(tf.square(new_image - train_y))
loss = lc + lf
# tf.add_to_collection('losses', loss)

# total_loss = tf.add_n(tf.get_collection('losses'))


train_step = tf.train.AdamOptimizer(0.001).minimize(loss)
sess = tf.Session()
init_op = tf.global_variables_initializer()
sess.run(init_op)
for i in range(steps):
    start = (i * batch_size) % dataset_size
    end = min(start + batch_size, dataset_size)
    sess.run(train_step,
             feed_dict={train_x: np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).repeat(batch_size, axis=0),
                        train_y: train_images[start:end]})

    if i % check_interval == 0 and i != 0:
        total_cross_entropy = sess.run(loss, feed_dict={
            train_x: np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).repeat(dataset_size, axis=0),
            train_y: train_images})

        a = sess.run(output_layer, feed_dict={
            train_x: np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).repeat(dataset_size, axis=0),
            train_y: train_images})
        b = sess.run(new_image, feed_dict={
            train_x: np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).repeat(dataset_size, axis=0),
            train_y: train_images})
        c = sess.run(model, feed_dict={
            train_x: np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).repeat(dataset_size, axis=0),
            train_y: train_images})

        print("After %d training step(s), loss on all data is %g" % (i, total_cross_entropy))
