# Module 2 Basic Tensorflow Operations
# Math Operations
# Author: Dr. Alfred Ang

import tensorflow as tf
sess = tf.Session()

# Step 1: Build the Graph
# a = tf.constant(1)
# b = tf.constant(2)
# c = tf.add(a,b)

# Step 2: Execute the Graph
# sess = tf.Session()
# print(sess.run(a))
# print(sess.run(b))
# print(sess.run(c))
#
# with tf.Session() as sess:
# 	print(sess.run(a))
# 	print(sess.run(b))
# 	print(sess.run(c))

# Data types

# Normally that is how you type Python
# a = 4
# b = 5.6
# print(a*b)

# a = tf.constant(4,dtype=tf.float32)
# b = tf.constant(5.6,dtype=tf.float32)
# c = tf.multiply(a,b)

# TF Math Operations
# a = tf.constant(2)
# b = tf.constant(3)
# c = tf.add(a,b)
# c = tf.subtract(a,b)
# c = tf.multiply(a,b)
# c = tf.div(a,b)
# c = tf.truediv(a, b)
# c = tf.floordiv(a, b)
# c = tf.mod(a, b)
# print(sess.run(c))

# Math Functions
# print(sess.run(tf.square(2)))
# print(sess.run(tf.sqrt(4.0)))
# print(sess.run(tf.sin(3.1416)))
# print(sess.run(tf.tan(3.1416)))
# print(sess.run(tf.cos(3.1416)))
# print(sess.run(tf.exp(1.0)))

# Exercise
# a = tf.constant(3,dtype=tf.float32)
# b = tf.constant(4.1,dtype=tf.float32)
# c = tf.constant(5,dtype=tf.float32)
# d = tf.add(tf.multiply(a,b),c)
#
# print(sess.run(d))

# Matrix
# a = tf.constant([
# 				[1,2,3],
# 				[4,5,6],
# 				[7,8,9]])
# print(sess.run(a))
# print(sess.run(tf.shape(a)))

# Tensor Shape and Slice
# b = tf.slice(a,[1,1],[2,2])
# print(sess.run(b))

# Basic Matrix Operations
# a = tf.constant([[1,2],[3,4]])
# b = tf.constant([[4,3],[2,1]])
# c = tf.add(a,b)
# c = tf.transpose(a)
# c = tf.matmul(a,b)
# print(sess.run(c))

# Exercise
# x = tf.constant([[1.1,1.2]],dtype=tf.float32)
# w = tf.constant([[1,2],[3,4]],dtype=tf.float32)
# b = tf.constant([[2.3,2.5]],dtype=tf.float32)
# y = tf.matmul(x,w)+b
# print(sess.run(y))

# Random Numbers
# tf.set_random_seed(2)
# a = tf.random_normal([2,3],mean=0,stddev=0.1)
# a = tf.truncated_normal([2,3],mean=0,stddev=0.1)
# print(sess.run(a))

# Comparison
# a = tf.constant(2)
# b = tf.constant(3)
# print(sess.run(tf.minimum(a,b)))
# print(sess.run(tf.maximum(a,b)))

# Exercise: Argmin and Argmax
# a = tf.constant([3,2,1,10,1,2,3])
# print(sess.run(tf.argmin(a)))
# print(sess.run(tf.argmax(a)))

# Ex: Reduced Sum
# a = tf.constant(
# 		[[1,1,1],
# 		 [2,2,2]]
#       )
# print(sess.run(tf.reduce_sum(a)))
# print(sess.run(tf.reduce_sum(a, 0)))
# print(sess.run(tf.reduce_sum(a, 1)))

# Special Matrices
# a = tf.zeros([2,3])
# a = tf.ones([2,3])
# a = tf.diag(np.ones(2))
# a = tf.fill([2,3],2)
# print(sess.run(a))

# Other Functions
# a = tf.linspace(-1., 1., 10)
# print(sess.run(a))

# Placeholder
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
