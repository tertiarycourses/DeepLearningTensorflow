# Module 9 Keras
# NN Model on MNIST dataset
# Author: Dr. Alfred Ang

import tensorflow as tf
tf.set_random_seed(25)

import keras
#from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense

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

L1 = 200
L2 = 100
L3 = 50

model = Sequential()
model.add(Dense(L1, input_dim=n_features, activation='relu'))
model.add(Dense(L2, activation='relu'))
model.add(Dense(L3, activation='relu'))
model.add(Dense(n_classes, activation='softmax'))
print(model.summary())

# Step 3: Compile the Model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Step 4: Train the Model
history = model.fit(X_train,y_train,
                    epochs=10,
                    batch_size = 100,
                    validation_data=(X_test,y_test))

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

import matplotlib.pyplot as plt
plt.plot(epochs,acc,'b',label='training accuracy')
plt.plot(epochs,val_acc,'r',label='testing accuracy')
plt.title('Training vs Testing Accuracy')
plt.xlabel('epoch')
plt.ylabel('Accuracy')
plt.legend()


plt.figure()
plt.plot(epochs,loss,'b',label='training loss')
plt.plot(epochs,val_loss,'r',label='testing loss')
plt.title('Training vs Testing Loss')
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

# Step 5: Evaluate the Model
loss,acc = model.evaluate(X_test, y_test)
print("\nTraining Accuracy = ",acc)

# Step 6: Save the Model
model.save("./models/mnist_nn.h5")

