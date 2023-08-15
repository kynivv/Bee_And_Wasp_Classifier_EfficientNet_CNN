import pandas as pd
import tensorflow as tf
import cv2
import numpy as np

import os
from tensorflow import keras
from keras import layers
from glob import glob
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint


# Hyperparameters
IMG_SIZE = 256
SPLIT = 0.25
EPOCHS = 10
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
BATCH_SIZE = 5


# Data Import
X = []
Y = []

data_dir = 'data/'

classes = os.listdir(data_dir)

for i, name in enumerate(classes):
    images = glob(f'{data_dir}/{name}/*.jpg')

    for image in images:
        img = cv2.imread(image)

        X.append(cv2.resize(img, (IMG_SIZE, IMG_SIZE)))
        Y.append(i)

X = np.asarray(X)
y = pd.get_dummies(Y).values


# Train Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size= SPLIT, random_state= 42)


# Model Based On EfficietNet
base_model = keras.applications.efficientnet.EfficientNetB3(include_top= False, weights= 'imagenet', input_shape= IMG_SHAPE, pooling= 'max')

model = keras.Sequential([
    base_model,
    layers.BatchNormalization(),
    layers.Dense(256, activation= 'relu'),
    layers.Dropout(rate= 0.3),
    layers.Dense(4, activation= 'softmax')
])

model.compile(optimizer= 'adam',
              loss= 'categorical_crossentropy',
              metrics= ['accuracy'],
              )


# Creating Callbacks
checkpoint = ModelCheckpoint('output/model_checkpoint.h5',
                             monitor= 'val_accuracy',
                             save_best_only= True,
                             save_weights_only= True,
                             verbose = 1
                             )


# Training The Model
history = model.fit(X_train, Y_train,
                    batch_size= BATCH_SIZE,
                    validation_data= (X_test, Y_test),
                    callbacks= checkpoint,
                    verbose= 1,
                    epochs= EPOCHS
                    )

