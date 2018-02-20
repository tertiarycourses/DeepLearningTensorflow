# Module 10 Appendix
# Custom TF Estimator

import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Step 1: Load Data
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('./mnist/', one_hot=True)

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": np.array(data.train.images)},
    y=np.argmax(data.train.labels, axis=1),
    num_epochs=None, shuffle=True)

test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": np.array(data.test.images)},
    y=np.argmax(data.test.labels, axis=1),
    num_epochs=1, shuffle=False)

# Step 2: Create Model

def model_fn(features, labels, mode, params):
    x = features["x"]
    net = tf.reshape(x, [-1, 28, 28, 1])
    net = tf.layers.conv2d(inputs=net, name='layer_conv1',filters=16, kernel_size=5,padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)
    net = tf.layers.conv2d(inputs=net, name='layer_conv2',filters=36, kernel_size=5,padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)
    net = tf.contrib.layers.flatten(net)
    net = tf.layers.dense(inputs=net, name='layer_fc1',units=128, activation=tf.nn.relu)
    net = tf.layers.dense(inputs=net, name='layer_fc2',units=10)

    logits = net
    y_pred = tf.nn.softmax(logits=logits)
    y_pred_cls = tf.argmax(y_pred, axis=1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        spec = tf.estimator.EstimatorSpec(mode=mode,predictions=y_pred_cls)
    else:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits)
        loss = tf.reduce_mean(cross_entropy)
        optimizer = tf.train.AdamOptimizer(learning_rate=params["learning_rate"])
        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())
        metrics = \
            {
                "accuracy": tf.metrics.accuracy(labels, y_pred_cls)
            }
        spec = tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=metrics)
    return spec

params = {"learning_rate": 1e-4}

feature_columns = [tf.feature_column.numeric_column("x", shape= (28,28))]
model = tf.estimator.Estimator(model_fn=model_fn,
                               params=params,
                               model_dir="./models/mnist-estimator-2/")

# Step 3: Train Model
model.train(input_fn=train_input_fn, steps=2000)

# Step 4: Evaluate Model
result = model.evaluate(input_fn=test_input_fn)
print("Classification accuracy: {0:.2%}".format(result["accuracy"]))