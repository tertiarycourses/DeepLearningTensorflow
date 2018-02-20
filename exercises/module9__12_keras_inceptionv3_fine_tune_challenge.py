# Module 9 Keras
# Challenge InceptionV3 Transfer Learning

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dense
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

train_data_dir = "./data/cats_dogs/"
img_width, img_height = 299, 299
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
base_model = InceptionV3(weights='imagenet', include_top=False)
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

# Step 2-2: Unfreeze and train the top 2 layers
for layer in model.layers[:172]:
    layer.trainable = False
for layer in model.layers[172:]:
    layer.trainable = True

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])
model.fit_generator(train_generator,steps_per_epoch=10,epochs=epochs)

# Save fine tuned weight
model.save('./models/inception_v3_cat_dog.h5')

