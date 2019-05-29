import tensorflow as tf
from numpy.random import RandomState

batch_size = 8
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

train_data = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
train_tag = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

a = tf.matmul(train_data, w1)
y = tf.matmul(a, w2)

y = tf.sigmoid(y)

cross_entropy = -tf.reduce_mean(train_tag * tf.log(tf.clip_by_value(y, 1e-10, 1.0))
                                + (1 - y) * tf.log(tf.clip_by_value(1 - y, 1e-10, 1.0)))

# cross_entropy = tf.reduce_mean(tf.square(train_tag - y))

train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
Y = [[int(x1 + x2 < 1)] for (x1, x2) in X]

with tf.Session() as session:
    init_op = tf.global_variables_initializer()
    session.run(init_op)

    steps = 5000
    for i in range(steps):
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)

        session.run(train_step, feed_dict={train_data: X[start:end], train_tag: Y[start:end]})

        if i % 1000 == 0:
            total_cross_entropy = session.run(cross_entropy, feed_dict={train_data: X, train_tag: Y})

            print("After %d training step(s), cross entropy on all data is %g" % (i, total_cross_entropy))
