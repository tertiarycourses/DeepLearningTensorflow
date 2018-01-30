# Module 9 Keras
# CNN Model on MNIST dataaset

from tensorflow.python.keras.models import load_model

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['TF_ENABLE_WINOGRAD_NONE_USED']='1'

# Step 1: Load the Model
model = load_model('./models/cifar_cnn.h5')

# Step 2: Evaluation

from PIL import Image
import numpy as np
img = Image.open('images/cat.jpg')
img = img.resize((32,32))
X_test = np.asarray(img)
X_test = np.expand_dims(X_test,axis=0)
prediction = model.predict(X_test)
print("Predicted Category Label : ", prediction.argmax(axis=1))





