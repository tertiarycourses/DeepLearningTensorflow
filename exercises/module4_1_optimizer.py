# Module 4: Machine Learning using Tensorflow
# Optimier
# Author: Dr. Alfred Ang

import tensorflow as tf

learn_rate = 0.1

x = tf.Variable(0.0)
y = tf.pow(x, 2) - 4.0 * x + 5.0

y_min = tf.train.AdamOptimizer(learn_rate).minimize(y)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for _ in range(100):
    _, y_val, x_val = sess.run([y_min, y, x])
    print('x:{}, :{}'.format(x_val, y_val))
