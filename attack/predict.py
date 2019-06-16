import tensorflow as tf
import numpy as np
import keras
from util import loader


def get_non_trainable_variable(input_variable):
    return tf.Variable(input_variable, trainable=False)


def compute_accuracy(session, prediction, v_xs, v_ys):
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(v_ys, 1))
    accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = session.run(accuracy_op, feed_dict={target_train_x: v_xs, target_train_label: v_ys})
    return result


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


sess = tf.Session()
# sess = tf_debug.LocalCLIDebugWrapperSession(sess)
image_width = 28
class_num = 10

fashion_mnist = keras.datasets.fashion_mnist
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
(train_images, train_labels), (test_images, test_labels) = loader.load_data()
train_images = (train_images / 255.0 - 0.5) * 2
test_images = (test_images / 255.0 - 0.5) * 2
train_labels = sess.run(tf.one_hot(train_labels, class_num))
test_labels = sess.run(tf.one_hot(test_labels, class_num))

target_train_x = tf.placeholder(tf.float32, shape=(None, image_width, image_width))
target_train_label = tf.placeholder(tf.float32, shape=(None, class_num))

# flatten_num = image_width ** 2
#
w1_t = tf.Variable(tf.random_normal([784, 128], stddev=1, seed=1))
w2_t = tf.Variable(tf.random_normal([128, 10], stddev=1, seed=1))
#
# target_output = tf.reshape(target_train_x, [-1, flatten_num])
# target_output = tf.matmul(target_output, w1_t)
# target_output = tf.nn.relu(target_output)
# target_output = tf.matmul(target_output, w2_t)
target_output = get_model(w1_t, w2_t, target_train_x)

# sparse不接受one-hot,不加sparse接受
# 这里还没有求均值，但是有负号
# target_output = tf.nn.softmax(target_output)
# cross_entropy = -tf.reduce_mean(target_train_label * tf.log(tf.clip_by_value(target_output, 1e-10, 1.0)))

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(labels=target_train_label, logits=target_output))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

dataset_size = len(train_images)

batch_size = 8
check_interval = 1000
steps = dataset_size // batch_size
steps = steps if dataset_size % batch_size == 0 else steps + 1

init_op = tf.global_variables_initializer()
sess.run(init_op)
epochs = 5

for epoch in range(epochs):
    print("Epoch %d / %d" % (epoch + 1, epochs))
    for i in range(steps):
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)

        sess.run(train_step,
                 feed_dict={target_train_x: train_images[start:end], target_train_label: train_labels[start:end]})

        if i % check_interval == 0 and i != 0:
            total_cross_entropy = sess.run(cross_entropy,
                                           feed_dict={target_train_x: train_images, target_train_label: train_labels})
            accuracy = compute_accuracy(sess, target_output, train_images, train_labels)
            print("After %d training step(s), loss: %g, accuracy: %g" % (i, total_cross_entropy, accuracy))
    total_cross_entropy = sess.run(cross_entropy,
                                   feed_dict={target_train_x: train_images, target_train_label: train_labels})
    accuracy = compute_accuracy(sess, target_output, train_images, train_labels)
    print("======================================================")
    print("At the end of epoch %d, loss: %g, accuracy: %g" % (epoch + 1, total_cross_entropy, accuracy))
    print("======================================================")

w1_n = sess.run(w1_t)
np.save('../model/nw1.npy', w1_n)
w2_n = sess.run(w2_t)
np.save('../model/nw2.npy', w2_n)
sess.close()
tf.reset_default_graph()

sess = tf.Session()
