# Module 3: Datasets
# Encoding and Decoding 1 Hot Vector
# Author: Dr. Alfred Ang

import numpy as np
import tensorflow as tf

# One Hot Encoding
# a = [0,1,2,1]
# num_labels = len(np.unique(a))
# b = np.eye(num_labels)[a]
# print(b)

# Ex: One Hot Decoding
a = tf.constant([[0,0,0,0,1,0,0,0,0,0]])
b = tf.argmax(a,axis=1)

sess = tf.Session()
print(sess.run(b))







