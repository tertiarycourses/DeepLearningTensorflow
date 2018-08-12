# Module 4: Machine Learning using Tensorflow
# Linear Regression
# Author: Dr. Alfred Ang

import tensorflow as tf
import numpy as np

# Step 1: Initial Setup

# Generate fictitious data
N=100
X_train = np.linspace(-10,10,N)
c1 = np.random.normal(loc=-0.5, scale=0.2, size=N)
c2 = np.random.normal(loc=1.0, scale=0.2, size=N)
y_train = c1 * X_train + c2

# Setup Variable
X = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
W = tf.Variable([0.1],tf.float32)
b = tf.Variable([0.1],tf.float32)

# Step 2: Model
yhat = tf.multiply(W,X) + b

# # Step 3: Loss Function
loss = tf.reduce_mean(tf.square(yhat - y))

# # Step 4: Optimizer
train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
# train = tf.train.AdagradDAOptimizer(0.01).minimize(loss)
# train = tf.train.AdadeltaOptimizer(0.01).minimize(loss)
# train = tf.train.AdamOptimizer(0.01).minimize(loss)
# train = tf.train.RMSPropOptimizer(0.01).minimize(loss)
# train = tf.train.MomentumOptimizer(0.01).minimize(loss)

# # training data
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# # Step 5: Training Loop
for i in range(1000):
  sess.run(train, {X:X_train, y:y_train})

# Step 6: Evaluation
W = sess.run(W)
b = sess.run(b)

print('W = {}, b={}'.format(W,b))

import matplotlib.pyplot as plt
plt.plot(X_train,y_train,'o')
plt.plot(X_train,W*X_train+b,'r')
plt.show()

