# Module 3: Datasets
# CIFAR-10 dataset
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# CIFAR-10 dataset from Keras
from tensorflow.python.keras.datasets import cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

import matplotlib.pyplot as plt
from scipy.misc import toimage
for i in range(0, 9):
	plt.subplot(3,3,i+1)
	plt.imshow(toimage(X_train[i]))
plt.show()
