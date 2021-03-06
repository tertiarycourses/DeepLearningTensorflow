# Module 9 Keras
# VGG16 Fine Tuning
# Author: Dr. Alfred Ang

from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dense
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

train_data_dir = "./images/cats_dogs/"
img_width, img_height = 224, 224
epochs = 1

# Data Augmentation
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
                        directory=train_data_dir,
                        target_size=[img_width, img_height],
                        class_mode='categorical')

# Step 2-1: Replace softmax Layer and add one dense layer
base_model = VGG16(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
prediction = Dense(2, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=prediction)
for layer in model.layers:
    layer.trainable = False
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])
model.fit_generator(train_generator,steps_per_epoch=10,epochs=epochs)

# Step 2-2: Unfreeze and train the top 5 layers
for layer in model.layers[:5]:
    layer.trainable = False
for layer in model.layers[5:]:
    layer.trainable = True

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])
model.fit_generator(train_generator,steps_per_epoch=10,epochs=epochs)

# Save fine tuned weight
model.save('./models/vgg16_cat_dog.h5')

