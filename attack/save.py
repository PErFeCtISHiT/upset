import os
import sys

sys.path.append('..')
import keras
import numpy as np
import tensorflow as tf
from util import loader
from attack import load_target_model
from attack import fashion_mnist_ssim
type_argv = sys.argv[1]
array = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
array[0][int(type_argv)] = 1


def get_model(w1, w2, x):
    flatten_num = image_width ** 2
    layer = tf.reshape(x, [-1, flatten_num])
    layer = tf.matmul(layer, w1)
    layer = tf.nn.relu(layer)
    layer = tf.matmul(layer, w2)
    layer = tf.nn.softmax(layer)
    return layer


image_width = 28
class_num = 10

fashion_mnist = keras.datasets.fashion_mnist
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
(train_images, train_labels), (test_images, test_labels) = loader.load_data()
train_images = (train_images / 255.0 - 0.5) * 2
test_images = (test_images / 255.0 - 0.5) * 2

dataset_size = len(train_images)

batch_size = 10
check_interval = 1000
steps = dataset_size // batch_size
steps = steps if dataset_size % batch_size == 0 else steps + 1

# t
train_x = tf.placeholder(tf.float32, shape=(None, 10), name='x-input')
# x
train_y = tf.placeholder(tf.float32, shape=(None, 28, 28), name='y-input')

arg_s = 0.7
arg_w = 1
arg_o = 0.5

current_layer = train_x
w1_upset = tf.Variable(tf.random_normal([10, 128], stddev=2, mean=0))
bias1 = tf.Variable(tf.constant(0.1, shape=[128]))
w2_upset = tf.Variable(tf.random_normal([128, 784], stddev=2, mean=0))
bias2 = tf.Variable(tf.constant(0.1, shape=[784]))
current_layer = tf.nn.relu(tf.layers.batch_normalization(tf.matmul(current_layer, w1_upset) + bias1))
current_layer = tf.tanh(tf.layers.batch_normalization(tf.matmul(current_layer, w2_upset) + bias2))
current_layer = tf.reshape(current_layer, [-1, 28, 28])
output_layer = current_layer

image_layer = tf.reshape(train_y, [-1, 784])
i1_upset = tf.Variable(tf.random_normal([784, 1024], stddev=2, mean=0))
i_bias1 = tf.Variable(tf.constant(0.1, shape=[1024]))
i2_upset = tf.Variable(tf.random_normal([1024, 784], stddev=2, mean=0))
i_bias2 = tf.Variable(tf.constant(0.1, shape=[784]))
image_layer = tf.nn.relu(tf.layers.batch_normalization(tf.matmul(image_layer, i1_upset) + i_bias1))
image_layer = tf.tanh(tf.layers.batch_normalization(tf.matmul(image_layer, i2_upset) + i_bias2))
image_layer = tf.reshape(image_layer, [-1, 28, 28])

new_image = tf.maximum(tf.minimum(arg_s * output_layer + image_layer * arg_o + train_y, 1), -1)
# new_image = tf.maximum(tf.minimum(arg_s * output_layer + train_y, 1), -1)
w1_n = tf.Variable(np.load('../model/w1.npy'), trainable=False)
w2_n = tf.Variable(np.load('../model/w2.npy'), trainable=False)

model = get_model(w1_n, w2_n, new_image)
#model = load_target_model.get_model_output(new_image)
lc = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(labels=train_x, logits=model))
# lf = arg_w * tf.reduce_mean(tf.square(new_image - train_y))
lf = - arg_w * tf.log(tf.clip_by_value(fashion_mnist_ssim.get_ssim_value_by_tensor(train_y, new_image), 1e-10, 1))
#lf = - arg_w * tf.reduce_mean(tf.log(
#    tf.clip_by_value(
#        tf.image.ssim(tf.reshape(train_y, [-1, 28, 28, 1]) / 2 + 0.5, tf.reshape(new_image, [-1, 28, 28, 1]) / 2 + 0.5,
#                      1.0) / 2 + 0.5, 1e-10, 1)))

loss = lc + lf

train_step = tf.train.AdamOptimizer(0.001).minimize(loss)
sess = tf.Session()
init_op = tf.global_variables_initializer()
sess.run(init_op)
epochs = 5
for epoch in range(epochs):
    if not os.path.isdir('../image/' + str(epoch)):
        os.mkdir('../image/' + str(epoch))
    print("Epoch %d / %d" % (epoch + 1, epochs))
    for i in range(steps):
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)
        sess.run(train_step,
                 feed_dict={train_x: array.repeat(batch_size, axis=0),
                            train_y: train_images[start:end]})

        if i % check_interval == 0 and i != 0:
            total_cross_entropy = sess.run(loss, feed_dict={
                train_x: array.repeat(dataset_size, axis=0),
                train_y: train_images})
            # b = sess.run(new_image, feed_dict={
            #     train_x: array.repeat(dataset_size, axis=0),
            #     train_y: train_images})
            # a = sess.run(model, feed_dict={
            #     train_x: array.repeat(dataset_size, axis=0),
            #     train_y: train_images})
            # c = sess.run(lc, feed_dict={
            #     train_x: array.repeat(dataset_size, axis=0),
            #     train_y: train_images})
            # ima = b[0]
            # ima = (ima / 2 + 0.5) * 255
            # im = Image.fromarray(ima)
            # im = im.convert('RGB')

            # im.save('../image/' + str(epoch) + '/' + str(i) + '.jpg')
            print("After %d training step(s), loss on all data is %g" % (i, total_cross_entropy))
    total_cross_entropy = sess.run(loss, feed_dict={
        train_x: array.repeat(dataset_size, axis=0),
        train_y: train_images})
    print("======================================================")
    print("At the end of epoch %d, loss: %g" % (epoch + 1, total_cross_entropy))
    print("======================================================")

w1_u = sess.run(w1_upset, feed_dict={
    train_x: array.repeat(dataset_size, axis=0),
    train_y: train_images})
w2_u = sess.run(w2_upset, feed_dict={
    train_x: array.repeat(dataset_size, axis=0),
    train_y: train_images})

np.save('../model/w1_u_' + str(type_argv) + '.npy', w1_u)
np.save('../model/w2_u_' + str(type_argv) + '.npy', w2_u)
