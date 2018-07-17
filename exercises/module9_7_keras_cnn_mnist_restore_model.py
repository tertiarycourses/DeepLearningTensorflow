# Module 9 Keras
# CNN Model on MNIST dataaset

from keras.models import load_model
import matplotlib.pyplot as plt


# Step 1: Load the Model
model = load_model('./models/mnist_cnn.h5')

# Step 2: Evaluation
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("mnist", one_hot=True)

X_train = mnist.train.images
y_train = mnist.train.labels
X_test = mnist.test.images
y_test = mnist.test.labels

import numpy as np
index = np.random.randint(1,10000)
X = X_test[index].reshape([1,28,28,1])
prediction = model.predict(X)
print("Predicted Digit : ", prediction.argmax(axis=1))

def show_digit(index):
    label = y_test[index].argmax(axis=0)
    image = X_test[index].reshape([28,28])
    plt.title('Actual Digit : {}'.format(label))
    plt.imshow(image, cmap='gray_r')
    plt.show()

show_digit(index)