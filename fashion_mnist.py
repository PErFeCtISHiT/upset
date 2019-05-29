import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

batch_size = 100

input_node = 784
output_node = 10

shadow_layer_node = 500
base_learn_rate = 0.8
learn_rate_decay = 0.99
regularization_rate = 0.0001
train_steps = 30000
moving_avg_decay = 0.99


def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    if avg_class is None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        return tf.matmul(layer1, weights2) + biases2
    else:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)


def train(mnist):
    train_data = tf.placeholder(tf.float32, [None, input_node], name='x-input')
    train_tag = tf.placeholder(tf.float32, [None, output_node], name='y-input')

    weights1 = tf.Variable(tf.truncated_normal([input_node, shadow_layer_node], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[shadow_layer_node]))

    weights2 = tf.Variable(tf.truncated_normal([shadow_layer_node, output_node], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[output_node]))

    # 不使用滑动平均
    y = inference(train_data, None, weights1, biases1, weights2, biases2)

    global_step = tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(moving_avg_decay, global_step)

    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # 使用滑动平均
    average_y = inference(train_data, variable_averages, weights1, biases1, weights2, biases2)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(train_tag, 1))

    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    regularizer = tf.contrib.layers.l2_regularizer(regularization_rate)

    regularization = regularizer(weights1) + regularizer(weights2)

    loss = cross_entropy_mean + regularization
    learning_rate = tf.train.exponential_decay(base_learn_rate, global_step, mnist.train.num_examples / batch_size,
                                               learn_rate_decay)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')
    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(train_tag, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as session:
        tf.global_variables_initializer().run()
        print(mnist.validation.labels.shape)
        validate_feed = {train_data: mnist.validation.images, train_tag: mnist.validation.labels}
        test_feed = {train_data: mnist.test.images, train_tag: mnist.test.labels}

        for i in range(train_steps):
            if i % 1000 == 0:
                temp_accuracy = session.run(accuracy, feed_dict=validate_feed)
                print("temp:" + str(temp_accuracy))

                t_x, t_y = mnist.train.next_batch(batch_size)
                session.run(train_op, feed_dict={train_data: t_x, train_tag: t_y})

        test_accuracy = session.run(accuracy, feed_dict=test_feed)
        print("final:" + str(test_accuracy))


def tag_format(tag):
    val = [[0 for col in range(len(tag))] for row in range(10)]
    print(val)
    for i in range(len(tag)):
        val[i][tag[i]] = 1
    print(val)
    return val




def main():
    fashion_mnist = input_data.read_data_sets("./data", one_hot=True)
    train(fashion_mnist)


main()
