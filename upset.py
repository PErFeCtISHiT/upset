import tensorflow as tf
import numpy as np
import keras
import loader

fashion_mnist = keras.datasets.fashion_mnist
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
(train_images, train_labels), (test_images, test_labels) = loader.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5)

for layer in model.layers:
    layer.trainable = False

# 上面是评估图片的模型，下面是upset模型
batch_size = 8

# t
train_x = tf.placeholder(tf.float32, shape=(None, 10), name='x-input')
# x
train_y = tf.placeholder(tf.float32, shape=(None, 784), name='y-input')

arg_s = 2
arg_w = 0.1


def flatten(images):
    return np.array(images).reshape(len(images), -1)


def get_weight(shape, lam):
    var = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
    # tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lam)(var))
    return var


layer_dimension = [10, 128, 256, 512, 1024, 512, 784]

num_layers = len(layer_dimension)
current_layer = train_x

in_dimension = layer_dimension[0]

for i in range(1, num_layers):
    out_dimension = layer_dimension[i]
    weight = get_weight([in_dimension, out_dimension], 0.001)
    bias = tf.Variable(tf.constant(0.1, shape=[out_dimension]))
    current_layer = tf.nn.relu((tf.matmul(current_layer, weight)))
    in_dimension = layer_dimension[i]

# x^
new_image = tf.maximum(tf.minimum(arg_s * current_layer + train_y, 1), -1)
lc = -tf.reduce_mean(train_x * tf.log(tf.clip_by_value(model.output, 1e-10, 1.0)))
lf = 0.08 * tf.reduce_mean(tf.square(new_image - train_y))
loss = lc + lf

# tf.add_to_collection('losses', loss)

# total_loss = tf.add_n(tf.get_collection('losses'))

train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

dataset_size = 50000

with tf.Session() as session:
    init_op = tf.global_variables_initializer()
    session.run(init_op)
    steps = 5000
    for i in range(steps):
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)

        session.run(train_step,
                    feed_dict={train_x: np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).repeat(batch_size, axis=0),
                               train_y: flatten(train_images[start:end])})

        if i % 1000 == 0:
            total_cross_entropy = session.run(loss, feed_dict={
                train_x: np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).repeat(batch_size, axis=0),
                train_y: flatten(train_images)})

            print("After %d training step(s), cross entropy on all data is %g" % (i, total_cross_entropy))
