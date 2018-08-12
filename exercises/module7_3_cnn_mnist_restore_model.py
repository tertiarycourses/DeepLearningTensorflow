# Module 7: Convolutional Neural Network (CNN)
# Restore CNN Model

import tensorflow as tf

# Step 1: Restore Graph
sess = tf.Session()
saver = tf.train.import_meta_graph('./models/mnist_cnn/mnist_cnn.ckpt.meta')
saver.restore(sess,tf.train.latest_checkpoint('./models/mnist_cnn'))


# Step 2: Restore Input and Output
graph = tf.get_default_graph()
X = graph.get_tensor_by_name("X:0")
yhat = graph.get_tensor_by_name("yhat:0")


# Step 3: Evaluation
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("mnist", one_hot=True,reshape=False,validation_size=0)

X_test = mnist.test.images
y_test = mnist.test.labels

import numpy as np
index = np.random.randint(1,10000)
X_test = X_test[index:index+1]

print("Actual answer : ",sess.run(tf.argmax(y_test[index])))
print("Predicted answer : ",sess.run(tf.argmax(yhat,1), feed_dict={X: X_test}))

