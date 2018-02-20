# Module 8: Recurrent Neural Network
# IMDB dataset

from keras.datasets import imdb
import numpy as np

max_words= 20000
max_len = 80

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_words)

# print(X_train)
# print(y_train)



