# Module 9 Keras
# RESNET Transfer Learning


import numpy as np

from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input,decode_predictions

# Step 1: Preprocess data
img = image.load_img("images/cobra.jpeg", target_size=(224, 224))

# Convert the image to a numpy array
img = image.img_to_array(img)

# Add a forth dimension since Keras expects a list of images
img = np.expand_dims(img, axis=0)

# Step 2: Load Pre-trained Model
model = ResNet50()

# Scale the input image to the range used in the trained network
img = preprocess_input(img)

# Run the image through the deep neural network to make a prediction
predictions = model.predict(img)

# Look up the names of the predicted classes. Index zero is the results for the first image.
predicted_classes = decode_predictions(predictions, top=3)

print("This is an image of:")

for imagenet_id, name, likelihood in predicted_classes[0]:
    print(" - {}: {:2f} likelihood".format(name, likelihood*100))

