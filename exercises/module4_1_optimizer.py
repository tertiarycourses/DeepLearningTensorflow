# Module 4: Machine Learning using Tensorflow
# Optimier
# Author: Dr. Alfred Ang

import tensorflow as tf

learn_rate = 0.1

x = tf.Variable(0.0)
loss = tf.pow(x, 2) - 4.0 * x + 5.0

optimizer = tf.train.AdamOptimizer(learn_rate).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for _ in range(100):
    _, loss_val, x_val = sess.run([optimizer, loss, x])
    print('x:{}, loss:{}'.format(x_val, loss_val))
