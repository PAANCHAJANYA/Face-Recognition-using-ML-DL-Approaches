from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Lambda
import tensorflow as tf
import glob
import os
import pickle
from mtcnn.mtcnn import MTCNN
from PIL import Image
import numpy as np
import cv2

TrainingImagePath='C:\\Users\\krish\\Desktop\\Krishna Work\\Face Detection and Recognition\\Face Datasets\\TrainDB\\Cropped FriendsDB'
train_datagen = ImageDataGenerator(shear_range=0.1,zoom_range=0.1,horizontal_flip=True,validation_split=0.3)
training_set = train_datagen.flow_from_directory(TrainingImagePath,target_size=(227,227),batch_size=64,class_mode='categorical',subset='training')
batchSize = 64
test_set = train_datagen.flow_from_directory(TrainingImagePath,target_size=(227, 227),class_mode='categorical',subset='validation')

TrainClasses=training_set.class_indices
ResultMap={}
for faceValue,faceName in zip(TrainClasses.values(),TrainClasses.keys()):
    ResultMap[faceValue]=faceName
with open("ResultsMap.pkl", 'wb') as fileWriteStream:
    pickle.dump(ResultMap, fileWriteStream)
print("Mapping of Face and its ID",ResultMap)



AlexNet = Sequential()

AlexNet.add(Conv2D(96, 11, strides=4, padding='same', input_shape=(227,227,3)))
AlexNet.add(Lambda(tf.nn.local_response_normalization))
AlexNet.add(Activation('relu'))
AlexNet.add(MaxPooling2D(3, strides=2))
AlexNet.add(Conv2D(256, 5, strides=4, padding='same'))
AlexNet.add(Lambda(tf.nn.local_response_normalization))
AlexNet.add(Activation('relu'))
AlexNet.add(MaxPooling2D(3, strides=2))
AlexNet.add(Conv2D(384, 3, strides=4, padding='same'))
AlexNet.add(Activation('relu'))
AlexNet.add(Conv2D(384, 3, strides=4, padding='same'))
AlexNet.add(Activation('relu'))
AlexNet.add(Conv2D(256, 3, strides=4, padding='same'))
AlexNet.add(Activation('relu'))
AlexNet.add(Flatten())
AlexNet.add(Dense(4096, activation='relu'))
AlexNet.add(Dropout(0.5))
AlexNet.add(Dense(4096, activation='relu'))
AlexNet.add(Dropout(0.5))
AlexNet.add(Dense(5, activation='softmax'))
AlexNet.summary()

AlexNet.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=["accuracy"])
AlexNet.fit(training_set, steps_per_epoch = training_set.samples // batchSize, validation_data=test_set, validation_steps = test_set.samples // batchSize, epochs=40)
AlexNet_json = AlexNet.to_json()
with open("AlexNet.json", "w") as classifier_file:
    classifier_file.write(AlexNet_json)
AlexNet.save_weights("AlexNet2.h5")
print("Saved the training model to disk")

