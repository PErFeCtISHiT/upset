import os

import keras
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau

import loader
import numpy as np

from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.optimizers import Adam

batch_data_path = '../model_data/batch_data.txt'
epoch_data_path = '../model_data/epoch_data.txt'
weight_path = '../model_data/my_model_weights.h5'


def ensure_pre_dirs_exists(*input_dirs):
    for dir_item in input_dirs:
        ensure_pre_dir_exists(dir_item)


def ensure_pre_dir_exists(input_dir):
    pre_dir = '/'.join(input_dir.split('/')[:-1])
    if not (os.path.exists(pre_dir) and os.path.isdir(pre_dir)):
        os.makedirs(pre_dir)


ensure_pre_dirs_exists(batch_data_path, epoch_data_path)


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
        file = open(batch_data_path, "a+")
        line = "accuracy: " + "\t" + str(logs.get('acc')) + "\t" + 'loss: ' + '\t' + \
               str(logs.get('loss')) + "\n"
        file.write(line)
        file.close()

    def on_epoch_end(self, batch, logs={}):
        file = open(epoch_data_path, "a+")
        line = "accuracy: " + "\t" + str(logs.get('acc')) + "\t" + 'loss: ' + '\t' + \
               str(logs.get('loss')) + "\t" + "val-accuracy: " + "\t" + \
               str(logs.get('val_acc')) + "\t" + 'val-loss: ' + '\t' + \
               str(logs.get('val_loss')) + "\n"
        file.write(line)
        file.close()


history = LossHistory()

sess = tf.Session()
# sess = tf_debug.LocalCLIDebugWrapperSession(sess)
image_width = 28
class_num = 10
channel_num = 1
batch_size = 64
epochs = 32

fashion_mnist = keras.datasets.fashion_mnist
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
(train_images, train_labels), (test_images, test_labels) = loader.load_data()
train_images = (train_images / 255.0 - 0.5) * 2
test_images = (test_images / 255.0 - 0.5) * 2
train_images = np.reshape(train_images, [-1, image_width, image_width, channel_num])
test_images = np.reshape(test_images, [-1, image_width, image_width, channel_num])
train_labels = sess.run(tf.one_hot(train_labels, class_num))
test_labels = sess.run(tf.one_hot(test_labels, class_num))

sess.close()
tf.reset_default_graph()

start = Input(shape=(image_width, image_width, channel_num,))
x = Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu', name="conv2d_1")(start)
# x = Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu', name="conv2d_2")(x)
x = MaxPool2D(pool_size=(2, 2), name="max2d_1")(x)
x = Dropout(0.25)(x)
# x = Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu', name="conv2d_3")(x)
# x = Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu', name="conv2d_4")(x)
# x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name="max2d_2")(x)
# x = Dropout(0.25)(x)
x = Flatten(name="flatten")(x)
x = Dense(256, activation='relu', name="dense1")(x)
x = Dropout(0.25)(x)
x = Dense(10, activation='softmax', name="dense2")(x)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3,
                                            verbose=1, factor=0.5, min_lr=0.00001)

model = Model(inputs=start, outputs=x)
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, validation_data=[test_images, test_labels], batch_size=batch_size, epochs=epochs,
          verbose=1,
          callbacks=[history, learning_rate_reduction])

model.save_weights(weight_path)
