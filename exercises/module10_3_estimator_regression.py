# Module 10 TF.Data and Estimator
# Estimator for Linear Regression
# Author: Dr. Alfred Ang

import numpy as np
import tensorflow as tf


# Step 1: Generate input points
N = 100
X_train = np.linspace(-10,10,N)
c1 = np.random.normal(loc=-0.5, scale=0.2, size=N)
c2 = np.random.normal(loc=1.0, scale=0.2, size=N)
y_train = c1 * X_train + c2

# Step 2: Create a feature column
x_col = tf.feature_column.numeric_column('x_coords')

# Step 3: Create a LinearRegressor Instance
regressor = tf.estimator.LinearRegressor([x_col])

# Step 4: Train the estimator with the generated data
train_input = tf.estimator.inputs.numpy_input_fn(
        x={'x_coords': X_train}, y=y_train,
        shuffle=True, num_epochs=1000)
regressor.train(train_input)

# Step 5: Predict the y-values when x equals 1.0 and 2.0
predict_input = tf.estimator.inputs.numpy_input_fn(
        x={'x_coords': np.array([1.0, 2.0], dtype=np.float32)},
        num_epochs=1, shuffle=False)
results = regressor.predict(predict_input)

for value in results:
    print(value['predictions'])