# Module 6: Tensorboard
# Tensorboard Demo
# Author: Dr. Alfred Ang

import tensorflow as tf

a = tf.constant(12,name='a')
b = tf.constant(4,name='b')
c = tf.multiply(a,b,name='c')
# d = tf.div(a, b, name='d')

# with tf.name_scope('multiply'):
#     c = tf.multiply(a, b, name='c')

# with tf.name_scope('divide'):
#     d = tf.div(a, b, name='d')

# Step 1: Convert to Tensorboard format
tf.summary.scalar('c',c)
#tf.summary.scalar('d',d)
merged_op = tf.summary.merge_all()

# Step 2: Specify the directory
file_writer = tf.summary.FileWriter('./tb/test1')


# Step 3: Run Session
sess = tf.Session()
c_val, summary = sess.run([c, merged_op])
#c_val,d_val, summary = sess.run([c,d, merged_op])
print(c_val)
#print(d_val)

# Step 4: Dump date to directory
file_writer.add_summary(summary)
file_writer.add_graph(sess.graph)
