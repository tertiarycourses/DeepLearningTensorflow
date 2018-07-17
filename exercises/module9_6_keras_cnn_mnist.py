# Module 9 Keras
# CNN Model on MNIST


from keras.models import Sequential
from keras.layers import Dense,Conv2D, MaxPooling2D,Flatten

# Parameters
n_classes = 10
learning_rate = 0.5
training_epochs = 1
batch_size = 100

# Step 1 Load the Data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("mnist", one_hot=True)

X_train = mnist.train.images.reshape(-1,28,28,1)
y_train = mnist.train.labels
X_test = mnist.test.images.reshape(-1,28,28,1)
y_test = mnist.test.labels

# Step 2: Build the Network
model = Sequential()
model.add(Conv2D(32,(3, 3),activation='relu',input_shape=(28,28,1),padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64,(3, 3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(n_classes, activation='softmax'))
#print(model.summary())

# Step 3: Compile the Model
model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])

# Step 4: Train the Model
model.fit(X_train, y_train, epochs=training_epochs, validation_data=[X_test,y_test])

# Step 5: Evaluate the Model
loss,acc = model.evaluate(X_test, y_test)
print('Test accuracy:', acc)

# Step 6: Save the Model
model.save("./models/mnist_cnn.h5")