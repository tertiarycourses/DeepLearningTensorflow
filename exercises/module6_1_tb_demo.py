# Module 6: Tensorboard
# Tensorboard Demo
# Author: Dr. Alfred Ang

import tensorflow as tf

a = tf.constant(12,name='a')
b = tf.constant(4,name='b')
# c = tf.multiply(a,b,name='c')
# d = tf.div(a, b, name='d')

with tf.name_scope('multiply'):
    c = tf.multiply(a, b, name='c')

with tf.name_scope('divide'):
    d = tf.div(a, b, name='d')

# Step 1: Generating Data Operations
tf.summary.scalar('c',c)
tf.summary.scalar('d',d)
merged_op = tf.summary.merge_all()

# Step 2: File Writer
file_writer = tf.summary.FileWriter('./tb/test1')

sess = tf.Session()

# Step 3: Run Operations in Session
c_val,d_val, summary = sess.run([c,d, merged_op])
print(c_val)
print(d_val)

# Step 4: Print Generated Date
file_writer.add_summary(summary)
file_writer.add_graph(sess.graph)
file_writer.flush()