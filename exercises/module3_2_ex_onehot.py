# Module 3: Datasets and Split Datasets

import numpy as np
import tensorflow as tf

# One Hot Encoding
# a = [0,1,2,1]
# num_labels = len(np.unique(a))
# b = np.eye(num_labels)[a]
# print(b)

# One Hot Decoding

a = tf.argmax([[0,1,0,0]],axis=1)

sess = tf.Session()
print(sess.run(a))







