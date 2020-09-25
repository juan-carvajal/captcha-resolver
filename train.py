import pickle
import os
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation
from keras.models import Sequential
import keras
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
cats = list('123456789ABCDEFHKMNPRSTUVWXYZ')


train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.0,
                                   zoom_range=0.0,
                                   horizontal_flip=False,
                                   validation_split=0.2)  # set validation split

img_width = 15
img_height = 30
batch_size = 32
train_data_dir = 'parts'
classes = len(cats)
nb_epochs = 3
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training')  # set as training data

validation_generator = train_datagen.flow_from_directory(
    train_data_dir,  # same directory as training data
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation')  # set as validation data

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(
    img_height, img_width, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=classes, activation='softmax'))


model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=nb_epochs)

model.save('model.h5')
