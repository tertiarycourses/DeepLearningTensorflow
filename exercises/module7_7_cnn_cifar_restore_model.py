# Module 7: Convolutional Neural Network (CNN)
# Restore CNN Model

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['TF_ENABLE_WINOGRAD_NONE_USED']='1'

# Step 1: Restore Graph
sess = tf.Session()
saver = tf.train.import_meta_graph('./models/cifar_cnn/cifar_cnn.ckpt.meta')
saver.restore(sess,tf.train.latest_checkpoint('./models/cifar_cnn'))

# Step 2: Restore Input and Output
graph = tf.get_default_graph()
X = graph.get_tensor_by_name("X:0")
yhat = graph.get_tensor_by_name("yhat:0")


# Step 3: Evaluation
from PIL import Image
import numpy as np
img = Image.open('images/car.jpg')
img = img.resize((32,32))
X_test = np.asarray(img)
X_test = np.expand_dims(X_test,axis=0)
prediction = sess.run(tf.argmax(yhat, 1), feed_dict={X: X_test})
print('Predicted Category Label : ', prediction)

