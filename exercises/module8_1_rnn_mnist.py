# Module 8: Recurrent Neural Network
# RNN model for MNIST dataset

import tensorflow as tf
from tensorflow.contrib import rnn

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
tf.set_random_seed(25)

# Parameters
learning_rate = 0.01
training_epochs = 1
batch_size = 28
rnn_size = 28

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("mnist", one_hot=True)

# Step 1: Initial Setup
X = tf.placeholder(tf.float32, [None, 28, 28])
y = tf.placeholder(tf.float32)

W = tf.Variable(tf.truncated_normal([rnn_size, 10],stddev=0.1))
B = tf.Variable(tf.truncated_normal([10],stddev=0.1))

# Step 2: Setup Model
inp = tf.unstack(X, axis=1)

cell = rnn.BasicLSTMCell(rnn_size)
H, C = rnn.static_rnn(cell, inp, dtype=tf.float32)

Ylogits = tf.matmul(H[-1], W) + B
yhat = tf.nn.softmax(Ylogits)

# Step 3: Loss Functions
loss = tf.reduce_mean(
   tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=Ylogits))

# Step 4: Optimizer
train = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# accuracy of the trained model, between 0 (worst) and 1 (best)
is_correct = tf.equal(tf.argmax(y,1),tf.argmax(yhat,1))
accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Step 5: Training Loop
for epoch in range(training_epochs):
    num_batches = int(mnist.train.num_examples / batch_size)
    for i in range(num_batches):
        batch_X, batch_y = mnist.train.next_batch(batch_size)
        batch_X = batch_X.reshape((batch_size, 28, 28))
        train_data = {X: batch_X, y: batch_y}
        sess.run(train, feed_dict=train_data)
        print(epoch * num_batches + i + 1, "Training accuracy =", sess.run(accuracy, feed_dict=train_data),
          "Loss =", sess.run(loss, feed_dict=train_data))

# Step 6: Evaluation
test_X = mnist.test.images
test_y = mnist.test.labels
test_X = test_X.reshape((-1, 28, 28))
print("Testing Accuracy = ", sess.run(accuracy, feed_dict = {X:test_X,y:test_y}))