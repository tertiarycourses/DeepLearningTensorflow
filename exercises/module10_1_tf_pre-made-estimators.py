# Module 10 Appendix
# TF Estimator

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
feature_columns = [tf.feature_column.numeric_column("x", shape= (28,28))]
model = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                   hidden_units=[512, 256, 128],
                                   activation_fn=tf.nn.relu,
                                   n_classes=10,
                                   model_dir="./models/mninst_estimator")

# Step 3: Train Model
model.train(input_fn=train_input_fn, steps=2000)

# Step 4: Evaluate Model
result = model.evaluate(input_fn=test_input_fn)
print("Classification accuracy: {0:.2%}".format(result["accuracy"]))