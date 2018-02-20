# Module 10: New Features in Tensorflow
# Eagle Execution

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
from tensorflow.contrib.eager.python import tfe

tfe.enable_eager_execution()
x = [[2.]]
m = tf.matmul(x,x)
print(m)

def square(x):
  return tf.multiply(x, x)

grad = tfe.gradients_function(square)

print(square(3.))
print(grad(3.))

gradgrad = tfe.gradients_function(lambda x: grad(x)[0])
print(gradgrad(3.))