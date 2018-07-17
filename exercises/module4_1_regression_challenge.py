# Module 4: Machine Learning using Tensorflow
# Simple TF Model - Linear Regression Challenge

import tensorflow as tf

# Step 1 Initial Setup
x = tf.placeholder(tf.float32)
W1 = tf.Variable([0.1],dtype=tf.float32)
W2 = tf.Variable([0.1],dtype=tf.float32)
b = tf.Variable([0.1],dtype=tf.float32)
y = tf.placeholder(tf.float32)

import numpy as np
X_train = np.linspace(-10.0,10.0,20)
y_train = 2*X_train*X_train - 1 + 10.0*np.random.randn(len(X_train))

# import matplotlib.pyplot as plt
# plt.scatter(X_train,y_train)
# plt.show()
#
# Step 2 Model

yhat = W1*x*x+W2*x+b

# Step 3 Loss Function

loss = tf.reduce_mean(tf.square(yhat-y))

# Step 4 Optimizer

train =tf.train.GradientDescentOptimizer(0.000001).minimize(loss)

# Step 5 Training Loop

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

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
