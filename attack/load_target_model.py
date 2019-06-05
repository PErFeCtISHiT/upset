import keras
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten

weight_path = '../model_data/my_model_weights.h5'


def get_model_output(input_tensor):
    input_tensor = tf.reshape(input_tensor, [-1, 28, 28, 1])
    start = Input(tensor=input_tensor)
    x = Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu', name="conv2d_1")(start)
    x = Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu', name="conv2d_2")(x)
    x = MaxPool2D(pool_size=(2, 2), name="max2d_1")(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu', name="conv2d_3")(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu', name="conv2d_4")(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name="max2d_2")(x)
    x = Flatten(name="flatten")(x)
    x = Dense(256, activation='relu', name="dense1")(x)
    x = Dense(10, activation='softmax', name="dense2")(x)
    model = Model(inputs=input_tensor, outputs=x)
    model.load_weights(weight_path, by_name=True)
    layers = model.layers
    for layer in layers:
        layer.trainable = False
    return model.output


test_tensor = tf.zeros([10, 28, 28, 1])
output = get_model_output(test_tensor)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(output))
