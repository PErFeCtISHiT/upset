import tensorflow as tf
import numpy as np
import keras
import loader


def get_non_trainable_variable(input_variable, session):
    return tf.Variable(session.run(input_variable), trainable=False)


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
epochs = 1

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

# 接下来是第二个模型
print()
print()

print('##################################################')
print('###################Second Model###################')
print('##################################################')

print()
print()

batch_size = 8
# t
train_x = tf.placeholder(tf.float32, shape=(None, 10), name='x-input')
# x
train_y = tf.placeholder(tf.float32, shape=(None, 28, 28), name='y-input')

arg_s = 2
arg_w = 0.1

layer_dimension = [10, 128, 256, 512, 1024, 512, 784]

num_layers = len(layer_dimension)
current_layer = train_x

in_dimension = layer_dimension[0]

for i in range(1, num_layers):
    out_dimension = layer_dimension[i]
    weight = get_weight([in_dimension, out_dimension])
    # bias = tf.Variable(tf.constant(0.1, shape=[out_dimension]))
    current_layer = tf.nn.relu((tf.matmul(current_layer, weight)))
    in_dimension = layer_dimension[i]

current_layer = tf.reshape(current_layer, [-1, 28, 28])

new_image = tf.maximum(tf.minimum(arg_s * current_layer + train_y, 1), -1)

w1_n = get_non_trainable_variable(w1_t, sess)
w2_n = get_non_trainable_variable(w2_t, sess)

model = get_model(w1_n, w2_n, train_y)

lc = -tf.reduce_mean(train_x * tf.log(tf.clip_by_value(model, 1e-10, 1.0)))
lf = 0.08 * tf.reduce_mean(tf.square(new_image - train_y))
loss = lc + lf

# tf.add_to_collection('losses', loss)

# total_loss = tf.add_n(tf.get_collection('losses'))

train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

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

        print("After %d training step(s), loss on all data is %g" % (i, total_cross_entropy))
