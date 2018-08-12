# Module 8: Recurrent Neural Network
# Challenge: RNN on IMDB dataset
# Author: Dr. Alfred Ang

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
tf.set_random_seed(25)

#Parameters
training_epochs = 2
batch_size = 100
learning_rate = 0.5

# Step 1: Pre-process data
from keras.preprocessing import sequence
from keras.datasets import imdb

max_words= 20000
max_len = 80
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_words)
X_train = sequence.pad_sequences(X_train,maxlen=max_len,padding='pre', truncating='pre')
X_test = sequence.pad_sequences(X_test,maxlen=max_len)

# One-hot encoding the labels
n_classes = len(np.unique(y_train)) # n_classes = 2
y_train = np.eye(n_classes)[y_train]
y_test = np.eye(n_classes)[y_test]

embedding_size = 128
rnn_size = 32
X = tf.placeholder('int32', [None, max_len])
y = tf.placeholder('int32')
W = tf.Variable(tf.truncated_normal([rnn_size, n_classes],stddev=0.1))
B = tf.Variable(tf.truncated_normal([n_classes],stddev=0.1))
embeddings = tf.Variable(tf.random_uniform([max_words, embedding_size], -1.0, 1.0))

# Step 2: Setup Model
x_embedded = tf.nn.embedding_lookup(embeddings, X)
x_embedded = tf.unstack(x_embedded, axis=1)
cell = rnn.BasicLSTMCell(rnn_size)
H, C = rnn.static_rnn(cell, x_embedded, dtype=tf.float32)

Ylogits = tf.matmul(H[-1], W) + B
yhat = tf.nn.softmax(Ylogits)

# Step 3: Loss Functions
loss = tf.reduce_mean(
   tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=Ylogits))

# Step 4: Optimizer
train = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

is_correct = tf.equal(tf.argmax(y,1),tf.argmax(yhat,1))
accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Step 5: Training Loop
for epoch in range(training_epochs):
    for i in range(int(X_train.shape[0] / batch_size)):
        batch_X = X_train[(i*batch_size):((i+1)*batch_size)]
        batch_y = y_train[(i*batch_size):((i+1)*batch_size)]
        train_data = {X: batch_X, y: batch_y}
        sess.run(train, feed_dict=train_data)
        print(epoch, "/", i,"Training Accuracy = ", sess.run(accuracy, feed_dict=train_data))

# Step 6: Evaluation
acc = []
for i in range(int(X_test.shape[0] / batch_size)):
    batch_X = X_test[(i*batch_size):((i+1)*batch_size)]
    batch_y = y_test[(i*batch_size):((i+1)*batch_size)]
    test_data = {X: batch_X, y: batch_y}
    sess.run(train, feed_dict = test_data)
    acc.append(sess.run(accuracy, feed_dict = test_data))

print("Testing Accuracy/Loss = ", sess.run(tf.reduce_mean(acc)))