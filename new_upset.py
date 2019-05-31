import tensorflow as tf
import numpy as np
import keras
import loader


def get_non_trainable_variable(input_variable, session):
    return tf.Variable(session.run(input_variable), trainable=False)


def compute_accuracy(session, prediction, v_xs, v_ys):
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={target_train_x: v_xs, target_train_label: v_ys})
    return result


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

target_train_x = tf.placeholder(tf.float32, shape=(None, image_width, image_width))
target_train_label = tf.placeholder(tf.int32, shape=(None, class_num))

flatten_num = image_width ** 2

w1_t = tf.Variable(tf.random_normal([784, 128], stddev=1, seed=1))
w2_t = tf.Variable(tf.random_normal([128, 10], stddev=1, seed=1))

target_output = tf.reshape(target_train_x, [-1, flatten_num])
target_output = tf.matmul(target_output, w1_t)
target_output = tf.nn.relu(target_output)
target_output = tf.matmul(target_output, w2_t)

# sparse不接受one-hot,不加sparse接受
# 这里还没有求均值，但是有负号
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(labels=target_train_label, logits=target_output))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

target_dataset_size = len(train_images)
target_batch_size = 8
check_interval = 1000
target_steps = target_dataset_size // target_batch_size
target_steps = target_steps if target_dataset_size % target_batch_size == 0 else target_steps + 1

init_op = tf.global_variables_initializer()
sess.run(init_op)
epochs = 10

for epoch in range(epochs):
    print("Epoch %d / %d" % (epoch + 1, epochs))
    for i in range(target_steps):
        start = (i * target_batch_size) % target_dataset_size
        end = min(start + target_batch_size, target_dataset_size)

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
    print("At the end of epoch %d, loss: %g, accuracy: %g" % (epoch, total_cross_entropy, accuracy))
    print("======================================================")
