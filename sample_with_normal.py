import tensorflow as tf
from numpy.random import RandomState

batch_size = 8

train_data = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
train_tag = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')


def get_weight(shape, lam):
    var = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lam)(var))
    return var


layer_dimension = [2, 3, 1]

num_layers = len(layer_dimension)
current_layer = train_data

in_dimension = layer_dimension[0]

for i in range(1, num_layers):
    out_dimension = layer_dimension[i]
    weight = get_weight([in_dimension, out_dimension], 0.001)
    bias = tf.Variable(tf.constant(0.1, shape=[out_dimension]))
    current_layer = (tf.matmul(current_layer, weight))
    in_dimension = layer_dimension[i]

current_layer = tf.sigmoid(current_layer)
loss_func = -tf.reduce_mean(train_tag * tf.log(tf.clip_by_value(current_layer, 1e-10, 1.0))
                            + (1 - current_layer) * tf.log(tf.clip_by_value(1 - current_layer, 1e-10, 1.0)))
tf.add_to_collection('losses', loss_func)

loss = tf.add_n(tf.get_collection('losses'))

train_step = tf.train.AdamOptimizer(0.001).minimize(loss_func)

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
            total_cross_entropy = session.run(loss_func, feed_dict={train_data: X, train_tag: Y})

            print("After %d training step(s), cross entropy on all data is %g" % (i, total_cross_entropy))
