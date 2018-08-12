# Module 4: Machine Learning using Tensorflow
# Simple TF Model - Linear Regression Challenge
# Author: Dr. Alfred Ang

import tensorflow as tf
import numpy as np

# Step 1 Initial Setup

# Generate fictitious data
N=100
X_train = np.linspace(-10,10,N)
c1 = np.random.normal(loc=2, scale=0.2, size=N)
c2 = np.random.normal(loc=1.0, scale=0.2, size=N)
c3 = np.random.normal(loc=-1.0, scale=0.2, size=N)
y_train = c1*X_train*X_train + c2*X_train + c3

x = tf.placeholder(tf.float32)
W1 = tf.Variable([0.1],dtype=tf.float32)
W2 = tf.Variable([0.1],dtype=tf.float32)
b = tf.Variable([0.1],dtype=tf.float32)
y = tf.placeholder(tf.float32)

# Step 2 Model
yhat = W1*x*x+W2*x+b

# Step 3 Loss Function
loss = tf.reduce_mean(tf.square(yhat-y))

# Step 4 Optimizer
train =tf.train.GradientDescentOptimizer(0.00001).minimize(loss)
# train =tf.train.AdamOptimizer(0.01).minimize(loss)

# Step 5 Training Loop
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(5000):
  sess.run(train,feed_dict={x:X_train,y:y_train})

# Step 6

W1 = sess.run(W1)
W2 = sess.run(W2)
b = sess.run(b)

import matplotlib.pyplot as plt
plt.plot(X_train,y_train,'o')
plt.plot(X_train,W1*X_train*X_train+W2*X_train+b,'r')
plt.show()
