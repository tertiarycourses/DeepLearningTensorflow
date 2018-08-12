import tensorflow as tf 
sess = tf.Session()

# Creating Dataset
# ds = tf.data.Dataset.range(10)

# Transforming Dataset
# ds_t = ds.filter(lambda x: x>2)
# ds_t = ds.map(lambda x: x*x)

# Loading Dataset
# iter = ds_t.make_one_shot_iterator()
#
# for _ in range(10):
#         print(sess.run(iter.get_next()))

# Create Dataset from Range
# ds = tf.data.Dataset.range(2,10)
# ds = tf.data.Dataset.range(2,10,2)

# Create Dataset from Tensor
# t1 = tf.constant([1,2])
# t2 = tf.constant([3,4])
# ds = tf.data.Dataset.from_tensors([t1,t2])

# # Create Dataset from each row of a tensor
# t1 = tf.constant([[1,2],[3,4]])
# ds = tf.data.Dataset.from_tensor_slices(t1)

# # Create Dataset from generator function
# def fgen():
#     x = 0
#     while x < 20:
#         yield x*x
#         x += 1
#
# ds = tf.data.Dataset.from_generator(fgen,output_types=tf.int64)
# iter = ds.make_one_shot_iterator()
#
# for _ in range(10):
#         print(sess.run(iter.get_next()))


# Dataset Operations
# ds = tf.data.Dataset.range(20)
# # ds2 = ds.take(5)
# ds2 = ds.skip(5)
# iter = ds2.make_one_shot_iterator()
#
# for _ in range(5):
#         print(sess.run(iter.get_next()))

ds1 = tf.data.Dataset.range(3)
ds2 = tf.data.Dataset.range(3,6)
ds3 = ds1.concatenate(ds2)
ds4 = ds3.batch(3)
ds5 = ds3.shuffle(6)

iter = ds5.make_one_shot_iterator()

for _ in range(6):
    print(sess.run(iter.get_next()))
