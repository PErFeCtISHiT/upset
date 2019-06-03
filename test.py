import tensorflow as tf
import numpy as np

y = tf.Variable(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]]), dtype=tf.float32)  # 每一行只有一个1
logits = tf.Variable(np.array([[12, 3, 2], [3, 10, 1], [1, 2, 5], [4, 6.5, 1.2], [3, 6, 1]]), dtype=tf.float32)
softmax_logits = tf.nn.softmax(logits)

sess = tf.Session()
init_op = tf.global_variables_initializer()
sess.run(init_op)
E1 = sess.run(-tf.reduce_mean(y * tf.log(tf.clip_by_value(softmax_logits, 1e-10, 1.0))))
E2 = sess.run(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits)))

print(E1)
print(E2)

# 可以看到上面的多除以了一个３
