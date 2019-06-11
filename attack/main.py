import numpy as np
import tensorflow as tf
import keras
from util import loader
import os
w1_upset = tf.Variable(np.load('../model/w1_u_1.npy'), trainable=False)
w2_upset = tf.Variable(np.load('../model/w2_u_1.npy'), trainable=False)

bias1 = tf.Variable(tf.constant(0.1, shape=[128]))
bias2 = tf.Variable(tf.constant(0.1, shape=[784]))

arg_s = 0.5
train_x = tf.placeholder(tf.float32, shape=(None, 10), name='x-input')
train_y = tf.placeholder(tf.float32, shape=(None, 28, 28), name='y-input')

current_layer = tf.nn.relu(tf.layers.batch_normalization(tf.matmul(train_x, w1_upset) + bias1))
current_layer = tf.tanh(tf.layers.batch_normalization(tf.matmul(current_layer, w2_upset) + bias2))
current_layer = tf.reshape(current_layer, [-1, 28, 28])
new_image_tensor = tf.maximum(tf.minimum(arg_s * current_layer + train_y, 1), -1)

sess = tf.Session()
init_op = tf.global_variables_initializer()
sess.run(init_op)


def compute_accuracy(session, old_estimate_class, new_estimate_class):
    return session.run(
        tf.reduce_mean(
            tf.cast(tf.not_equal(tf.argmax(old_estimate_class, 1), tf.argmax(new_estimate_class, 1)), tf.float32)))


def get_model(w1, w2, x):
    layer = tf.reshape(x, [-1, 784])
    layer = tf.matmul(layer, w1)
    layer = tf.nn.relu(layer)
    layer = tf.matmul(layer, w2)
    layer = tf.nn.softmax(layer)
    return layer


def aiTest(images, shape):
    global new_image
    global ssim_accuracy
    images = (images / 255.0 - 0.5) * 2
    # for i in range(10):
    #     t = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    #     t[0][i % 10] = 1
    #     new_image = sess.run(new_image_tensor, feed_dict={train_x: t.repeat(shape[0], axis=0), train_y: images})
    t = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]])
    ssim_accuracy2 = tf.reduce_mean(tf.image.ssim(tf.reshape(train_y, [-1, 28, 28, 1]) / 2 + 0.5,
                                                  tf.reshape(new_image_tensor, [-1, 28, 28, 1]) / 2 + 0.5,
                                                  1.0) / 2 + 0.5)
    # ssim_accuracy1 = tf.image.ssim(train_y / 2 + 0.5, new_image_tensor / 2 + 0.5, 1)
    # a = sess.run(ssim_accuracy1, feed_dict={train_x: t.repeat(shape[0], axis=0), train_y: images})
    ssim_accuracy = sess.run(ssim_accuracy2, feed_dict={train_x: t.repeat(shape[0], axis=0), train_y: images})
    new_image = sess.run(new_image_tensor, feed_dict={train_x: t.repeat(shape[0], axis=0), train_y: images})
    generate_images = (new_image / 2 + 0.5) * 255
    return generate_images


fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = loader.load_data()

new_images = aiTest(test_images[0:1000], [1000, 28, 28, 1])

w1_n = tf.Variable(np.load('../model/w1.npy'), trainable=False)
w2_n = tf.Variable(np.load('../model/w2.npy'), trainable=False)

init_op = tf.global_variables_initializer()
sess.run(init_op)

target_train_x = tf.placeholder(tf.float32, shape=(None, 28, 28))
old_model = get_model(w1_n, w2_n, target_train_x)

old_estimate_class = sess.run(old_model, feed_dict={target_train_x: test_images[0:1000]})
new_estimate_class = sess.run(old_model, feed_dict={target_train_x: new_images})

# for index in range(len(old_estimate_class)):
#     print("The old is ", old_estimate_class[index], "and the new is ", new_estimate_class[index])
#     print()

accuracy = compute_accuracy(sess, old_estimate_class, new_estimate_class)

print('Attack Accuracy is %g' % accuracy)

# ssim_accuracy = fashion_mnist_ssim.get_ssim_value(test_images[0:1000], new_images)
print('ssim accuracy is %g' % ssim_accuracy)
# for pic_class in b:
#     print(pic_class)

if not os.path.isdir('../image/test'):
    os.mkdir('../image/test')
# for j in range(len(new_images)):
#     ima = new_images[j]
#     im = Image.fromarray(ima)
#     im = im.convert('RGB')
#     im.save('../image/test/' + str(j) + '.jpg')
