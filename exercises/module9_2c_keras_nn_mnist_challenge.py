# Module 9 Keras
# NN Model on MNIST dataset Challenge

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Hyper Parameters
n_features = 784
n_classes = 10
learning_rate = 0.5
training_epochs = 2

# Step 1: Pre-process the  Data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("mnist", one_hot=True)

X_train = mnist.train.images
y_train = mnist.train.labels
X_test = mnist.test.images
y_test = mnist.test.labels

# Step 2: Define the Model

L1 = 1024
L2 = 512
L3 = 256
L4 = 128
L5 = 64
L6 = 32

model = Sequential()
model.add(Dense(L1, input_dim=n_features, activation='relu'))
model.add(Dense(L2, activation='relu'))
model.add(Dense(L3, activation='relu'))
model.add(Dense(L4, activation='relu'))
model.add(Dense(L5, activation='relu'))
model.add(Dense(L6, activation='relu'))
model.add(Dense(n_classes, activation='softmax'))

print(model.summary())

# Step 3: Compile the Model
model.compile(optimizer='adam',loss='categorical_crossentropy',
              metrics=['accuracy'])

# Step 4: Train Model
model.fit(X_train, y_train,
          epochs=training_epochs,
          validation_data=(X_test, y_test),
          shuffle=True)

# Step 5: Evaluation
score = model.evaluate(X_test, y_test)
print("\nTraining Accuracy = ",score[1],"Loss",score[0])

