# Module 9 Keras
# RNN Model on IMDB dataaset

import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

max_features = 20000
maxlen = 80

# Step 1: Pre-process the data
from keras.datasets import imdb
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)

X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)

# n_classes = len(np.unique(y_train)) # n_classes = 2
# y_train = np.eye(n_classes)[y_train]
# y_test = np.eye(n_classes)[y_test]

# Step 2: Build the Model
model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(128,activation='tanh'))
# model.add(Dense(2,activation='softmax'))
model.add(Dense(1,activation='sigmoid'))

# Step 3: Compile the Model
model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])

# Step 4: Train the Model
model.fit(X_train, y_train, epochs=2)

# Step 5: Evaluate the Model
loss,acc = model.evaluate(X_test, y_test)
print('Test accuracy:', acc)