# Module 9 Keras
# RNN Model on MNIST dataaset

import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, SimpleRNN, LSTM,GRU

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

n_classes = 10
epochs = 2
rnn_units = 28

# Step 1 Preprocess data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("mnist", one_hot=True)

X_train = mnist.train.images.reshape(-1,28,28)
y_train = mnist.train.labels
X_test = mnist.test.images.reshape(-1,28,28)
y_test = mnist.test.labels

# Step 2 Create the Model
model = Sequential()

model.add(LSTM(rnn_units, activation='tanh', input_shape=[28,28]))
model.add(Dense(n_classes, activation='softmax'))

# Step 3: Compile the Model
model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])

# Step 4: Train the Model
model.fit(X_train, y_train,epochs=epochs,)

# Step 5: Evaluate the Model
score = model.evaluate(X_test, y_test)
print('Test accuracy:', score[1])