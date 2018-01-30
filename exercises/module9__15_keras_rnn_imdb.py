# Module 9 Keras
# RNN Model on IMDB dataaset

from tensorflow.python import keras
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Embedding, LSTM
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Parameters
max_features = 20000
maxlen = 80  # cut texts after this number of words

# Step 1: Pre-process the data
from tensorflow.python.keras.datasets import imdb
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)
# print(len(X_train), 'train sequences')
# print(len(X_test), 'test sequences')

X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
# print('X_train shape:', X_train.shape)
# print('X_test shape:', X_test.shape)

# Step 2: Build the Model
model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(128,activation='tanh'))
model.add(Dense(1,activation='sigmoid'))

# Step 3: Compile the Model
model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])

# Step 4: Train the Model
model.fit(X_train, y_train, epochs=2)

# Step 5: Evaluate the Model
score = model.evaluate(X_test, y_test)
print('Test accuracy:', score[1])