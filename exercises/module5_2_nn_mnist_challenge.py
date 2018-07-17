# Module 5: Neural Network and Deep Learning
# NN model for MNIST dataset and save model

import tensorflow as tf

# Hyper Parameters
learning_rate = 0.01
training_epochs = 2
batch_size = 100
tf.set_random_seed(25)

# Step 1: Initial Setup
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("mnist", one_hot=True)

X = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

L1 = 200
L2 = 100
L3 = 50
L4 = 40
L5 = 30
L6 = 20

X = tf.placeholder(tf.float32,[None,784],name="X")
y = tf.placeholder(tf.float32,[None,10])
W1 = tf.Variable(tf.truncated_normal([784,L1],stddev=0.1))
b1 = tf.Variable(tf.truncated_normal([L1],stddev=0.1))
W2 = tf.Variable(tf.truncated_normal([L1,L2],stddev=0.1))
b2 = tf.Variable(tf.truncated_normal([L2],stddev=0.1))
W3 = tf.Variable(tf.truncated_normal([L2,L3],stddev=0.1))
b3 = tf.Variable(tf.truncated_normal([L3],stddev=0.1))
W4 = tf.Variable(tf.truncated_normal([L3,L4],stddev=0.1))
b4 = tf.Variable(tf.truncated_normal([L4],stddev=0.1))
W5 = tf.Variable(tf.truncated_normal([L4,L5],stddev=0.1))
b5 = tf.Variable(tf.truncated_normal([L5],stddev=0.1))
W6 = tf.Variable(tf.truncated_normal([L5,L6],stddev=0.1))
b6 = tf.Variable(tf.truncated_normal([L6],stddev=0.1))
W7 = tf.Variable(tf.truncated_normal([L6,10],stddev=0.1))
b7 = tf.Variable(tf.truncated_normal([10],stddev=0.1))

# Step 2: Setup Model
# Y1 = tf.nn.sigmoid(tf.matmul(X, W1) + B1)
# Y2 = tf.nn.sigmoid(tf.matmul(Y1, W2) + B2)
# Y3 = tf.nn.sigmoid(tf.matmul(Y2, W3) + B3)
# Y4 = tf.nn.sigmoid(tf.matmul(Y3, W4) + B4)
Y1 = tf.nn.relu(tf.matmul(X,W1)+b1)
Y2 = tf.nn.relu(tf.matmul(Y1,W2)+b2)
Y3 = tf.nn.relu(tf.matmul(Y2,W3)+b3)
Y4 = tf.nn.relu(tf.matmul(Y3,W4)+b4)
Y5 = tf.nn.relu(tf.matmul(Y4,W5)+b5)
Y6 = tf.nn.relu(tf.matmul(Y5,W6)+b6)
Ylogits = tf.matmul(Y6,W7)+b7
yhat = tf.nn.softmax(Ylogits,name="yhat")

# Step 3: Loss Functions
loss = tf.reduce_mean(
   tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=Ylogits))

# Step 4: Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

# accuracy of the trained model, between 0 (worst) and 1 (best)
is_correct = tf.equal(tf.argmax(y,1),tf.argmax(yhat,1))
accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Step 5: Train the Model
for epoch in range(training_epochs):
    num_batches = int(mnist.train.num_examples/batch_size)
    for i in range(num_batches):
        batch_X, batch_y = mnist.train.next_batch(batch_size)
        train_data = {X: batch_X, y: batch_y}
        sess.run(train, feed_dict=train_data)

        print(epoch*num_batches+i+1, "Training accuracy =", sess.run(accuracy, feed_dict=train_data),
              "Loss =", sess.run(loss, feed_dict=train_data))

# Step 6: Evaluate the Model
test_data = {X:mnist.test.images,y:mnist.test.labels}
print("Testing Accuracy = ", sess.run(accuracy, feed_dict = test_data))

