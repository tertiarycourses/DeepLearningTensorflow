# Module 2 Basic Tensorflow Operations
# Placeholders
# Author: Dr. Alfred Ang

import tensorflow as tf
sess = tf.Session()

# a = tf.placeholder(tf.float32)
# b = tf.placeholder(tf.float32)
# sum = tf.add(a,b)
# print(sess.run(sum,{a:3,b:4}))
# print(sess.run(sum,feed_dict={a:3,b:4}))

# Challenge

# a = tf.placeholder(tf.float32,shape=[1,2])
# w = tf.placeholder(tf.float32,shape=[2,2])
# b = tf.constant([[3.,3.]],tf.float32)
# y = tf.add(tf.matmul(a,w),b)
# print(sess.run(y,feed_dict={
#     a:[[1,1]],
#     w:[[1,2],[3,4]]})
#       )
#
# x = tf.placeholder(tf.float32,shape=(2,2))
# y = tf.add(tf.matmul(x,x),x)
# print(sess.run(y,feed_dict={x:[[1,1],[1,1]]}))

# Variables
# W = tf.Variable([1.], tf.float32)
# b = tf.Variable([-1.], tf.float32)
# x = tf.placeholder(tf.float32)
# y = W * x + b
#
# init = tf.global_variables_initializer()
# sess.run(init)
# print(sess.run(y, {x:[1,2,3,4]}))