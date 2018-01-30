# Module 9 Keras
# Tensorboard Challenge

from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Hyper Parameters
n_features = 784
n_classes = 10
learning_rate = 0.5
training_epochs = 5

# Step 1: Pre-process the  Data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("mnist", one_hot=True)

X_train = mnist.train.images
y_train = mnist.train.labels
X_test = mnist.test.images
y_test = mnist.test.labels

# Step 2: Build the  Network
L1 = 200
L2 = 100
L3 = 60
L4 = 30

model = Sequential()
model.add(Dense(L1, input_dim=n_features, activation='relu',name='layer_1'))
# model.add(Dense(L2, activation='sigmoid',name='layer_2'))
# model.add(Dense(L3, activation='sigmoid',name='layer_3'))
# model.add(Dense(L4, activation='sigmoid',name='layer_4'))
model.add(Dense(L2, activation='relu',name='layer_2'))
model.add(Dense(L3, activation='relu',name='layer_3'))
model.add(Dense(L4, activation='relu',name='layer_4'))
model.add(Dense(n_classes, activation='softmax',name='output_layer'))

# Step 3: Compile the Model
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

# Step 4: Training

# Create a TensorBoard logger
logger = keras.callbacks.TensorBoard(
    log_dir='viz/relu',
    histogram_freq=1,
    write_graph=True,
    write_images=False
)

model.fit(X_train,y_train,
          epochs=training_epochs,
          validation_data=[X_test,y_test],
          callbacks=[logger])

# Step 5: Evaluation
score = model.evaluate(X_test, y_test)
print("\nTraining Accuracy = ",score[1],"Loss",score[0])


