import numpy as np
import tensorflow as tf
import keras
import loader
from PIL import Image

w1_upset = tf.convert_to_tensor(np.load('w1_u.npy'))
w2_upset = tf.convert_to_tensor(np.load('w2_u.npy'))

bias1 = tf.Variable(tf.constant(0.1, shape=[128]))
bias2 = tf.Variable(tf.constant(0.1, shape=[784]))

arg_s = 1

train_x = tf.placeholder(tf.float32, shape=(None, 10), name='x-input')
train_y = tf.placeholder(tf.float32, shape=(None, 28, 28), name='y-input')

current_layer = tf.nn.relu(tf.layers.batch_normalization(tf.matmul(train_x, w1_upset) + bias1))
current_layer = tf.tanh(tf.layers.batch_normalization(tf.matmul(current_layer, w2_upset) + bias2))
current_layer = tf.reshape(current_layer, [-1, 28, 28])
new_image_tensor = tf.maximum(tf.minimum(arg_s * current_layer + train_y, 1), -1)

sess = tf.Session()
init_op = tf.global_variables_initializer()
sess.run(init_op)


def get_model(w1, w2, x):
    layer = tf.reshape(x, [-1, 784])
    layer = tf.matmul(layer, w1)
    layer = tf.nn.relu(layer)
    layer = tf.matmul(layer, w2)
    layer = tf.nn.softmax(layer)
    return layer


def aiTest(images, shape):
    global new_image
    images = (images / 255.0 - 0.5) * 2
    for i in range(10):
        t = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        t[0][i % 10] = 1
        new_image = sess.run(new_image_tensor, feed_dict={train_x: t.repeat(shape[0], axis=0), train_y: images})
    generate_images = (new_image / 2 + 0.5) * 255
    return generate_images


fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = loader.load_data()

new_images = aiTest(test_images[0:1000], [1000, 28, 28, 1])

w1_n = tf.convert_to_tensor(np.load('w1.npy'))
w2_n = tf.convert_to_tensor(np.load('w2.npy'))

model = get_model(w1_n, w2_n, new_images)
t = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
b = sess.run(model, feed_dict={train_x: t.repeat(1000, axis=0), train_y: test_images[0:1000]})
for j in range(len(new_images)):
    ima = new_images[j]
    im = Image.fromarray(ima)
    im = im.convert('RGB')
    im.save('image/test' + str(j) + '.jpg')
