# Module 9 Keras
# Restore Iris Model

from tensorflow.python.keras.models import load_model

# Step 1: Restore Model
model = load_model('./models/iris.h5')

# Step 2: Evaluate
import numpy as np
X = [3.1,2.1,4.1,5.5]
X = np.reshape(X,[-1,4])

flower = {0:"sentosa",1:"vicolor",2:"virgica"}

prediction = np.argmax(model.predict(X))

print("This flower is ", flower[prediction])



