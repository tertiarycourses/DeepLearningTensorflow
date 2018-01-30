# Module 7: Convolutional Neural Network (CNN)
# Challenge : CIFAR-10 dataset

import tensorflow as tf
from tensorflow.python import keras
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_WINOGRAD_NONE_USED']='1'

# Parameters
learning_rate = 0.01
training_epochs = 1
batch_size = 100

# Step 1: Initial Setup
from tensorflow.python.keras.datasets import cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

X = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.float32, [None, 10])
pkeep = tf.placeholder(tf.float32)

L1 = 4  # first convolutional layer output depth
L2 = 8  # second convolutional layer output depth
L3 = 16  # Fully connected layer

W1 = tf.Variable(tf.truncated_normal([3, 3, 3, L1], stddev=0.1))
B1 = tf.Variable(tf.truncated_normal([L1],stddev=0.1))
W2 = tf.Variable(tf.truncated_normal([3, 3, L1, L2], stddev=0.1))
B2 = tf.Variable(tf.truncated_normal([L2],stddev=0.1))
W3 = tf.Variable(tf.truncated_normal([8 * 8 * L2, L3], stddev=0.1))
B3 = tf.Variable(tf.truncated_normal([L3],stddev=0.1))
W4 = tf.Variable(tf.truncated_normal([L3, 10], stddev=0.1))
B4 = tf.Variable(tf.truncated_normal([10],stddev=0.1))

# Step 2: Setup Model
Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME') + B1)
Y1 = tf.nn.max_pool(Y1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, 1, 1, 1], padding='SAME') + B2)
Y2 = tf.nn.max_pool(Y2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

YY = tf.reshape(Y2, shape=[-1, 8 * 8 * L2])

Y3 = tf.nn.relu(tf.matmul(YY, W3) + B3)
#YY3 = tf.nn.dropout(Y3, pkeep)
Ylogits = tf.matmul(Y3, W4) + B4
yhat = tf.nn.softmax(Ylogits)

# Step 3: Loss Functions
loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(logits=Ylogits, labels=y))

# Step 4: Optimizer
train = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# accuracy of the trained model, between 0 (worst) and 1 (best)
is_correct = tf.equal(tf.argmax(y, 1), tf.argmax(yhat, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Step 5: Training Loop
for epoch in range(training_epochs):
    num_batches = int(X_train.shape[0] / batch_size)
    for i in range(num_batches):
        batch_X = X_train[(i*batch_size):((i+1)*batch_size)]
        batch_y = y_train[(i*batch_size):((i+1)*batch_size)]
        train_data = {X: batch_X, y: batch_y}
        sess.run(train, feed_dict = train_data)
        print(epoch * num_batches + i + 1,
              "Training accuracy =", sess.run(accuracy, feed_dict=train_data),
              "Loss =", sess.run(loss, feed_dict=train_data))

# Step 6: Evaluation
acc = []
for i in range(int(X_test.shape[0] / batch_size)):
    batch_X = X_test[(i*batch_size):((i+1)*batch_size)]
    batch_y = y_test[(i*batch_size):((i+1)*batch_size)]
    test_data = {X: batch_X, y: batch_y}
    sess.run(train, feed_dict = test_data)
    acc.append(sess.run(accuracy, feed_dict = test_data))

print("Testing Accuracy/Loss = ", sess.run(tf.reduce_mean(acc)))