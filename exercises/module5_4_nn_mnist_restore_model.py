# Module 4: Simple TF Model
# Use softmax cross-entropy function on MINST dataset

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Step 1: Restore Graph
sess = tf.Session()
saver = tf.train.import_meta_graph('./models/mnist_nn/mnist_nn.ckpt.meta')
saver.restore(sess,tf.train.latest_checkpoint('./models/mnist_nn'))

# Step 2: Restore Input and Output
graph = tf.get_default_graph()
X = graph.get_tensor_by_name("X:0")
yhat = graph.get_tensor_by_name("yhat:0")


# Step 3: Evaluation
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("mnist", one_hot=True)

X_test = mnist.test.images
y_test = mnist.test.labels

import numpy as np
index = np.random.randint(1,10000)
X_test = X_test[index:index+1].reshape([-1,784])

print("Actual answer : ",sess.run(tf.argmax(y_test[index])))
print("Predicted answer : ",sess.run(tf.argmax(yhat,1), feed_dict={X: X_test}))
