# Module 7: Convolutional Neural Network (CNN)
# Droptout
# Author: Dr. Alfred Ang


import tensorflow as tf

X = [1.5, 0.5, 0.75, 1.0, 0.75, 0.6]
p_keep = 0.5
drop_out = tf.nn.dropout(X, p_keep)

sess = tf.Session()
#print(sess.run(tf.multiply(1/p_keep,X)))
print(sess.run(drop_out))
sess.close()