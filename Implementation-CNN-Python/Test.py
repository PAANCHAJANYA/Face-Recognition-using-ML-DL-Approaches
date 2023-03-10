from keras.models import model_from_json
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Lambda
from tensorflow.keras import backend as K
from keras.preprocessing import image
from mtcnn.mtcnn import MTCNN
import numpy as np
from os import listdir
import pickle
import os
import cv2

AlexNet_loaded_model = Sequential()

AlexNet_loaded_model.add(Conv2D(96, 11, strides=4, padding='same', input_shape=(227,227,3)))
AlexNet_loaded_model.add(Lambda(tf.nn.local_response_normalization))
AlexNet_loaded_model.add(Activation('relu'))
AlexNet_loaded_model.add(MaxPooling2D(3, strides=2))
AlexNet_loaded_model.add(Conv2D(256, 5, strides=4, padding='same'))
AlexNet_loaded_model.add(Lambda(tf.nn.local_response_normalization))
AlexNet_loaded_model.add(Activation('relu'))
AlexNet_loaded_model.add(MaxPooling2D(3, strides=2))
AlexNet_loaded_model.add(Conv2D(384, 3, strides=4, padding='same'))
AlexNet_loaded_model.add(Activation('relu'))
AlexNet_loaded_model.add(Conv2D(384, 3, strides=4, padding='same'))
AlexNet_loaded_model.add(Activation('relu'))
AlexNet_loaded_model.add(Conv2D(256, 3, strides=4, padding='same'))
AlexNet_loaded_model.add(Activation('relu'))
AlexNet_loaded_model.add(Flatten())
AlexNet_loaded_model.add(Dense(4096, activation='relu'))
AlexNet_loaded_model.add(Dropout(0.5))
AlexNet_loaded_model.add(Dense(4096, activation='relu'))
AlexNet_loaded_model.add(Dropout(0.5))
AlexNet_loaded_model.add(Dense(5, activation='softmax'))
AlexNet_loaded_model.summary()
AlexNet_loaded_model.load_weights("AlexNet3.h5")
with open('ResultsMap.pkl', 'rb') as f:
    ResultMap = pickle.load(f)
dirName = "C:\\Users\\krish\\Desktop\\Krishna Work\\Face Detection and Recognition\\Face Datasets\\TestDB\\FriendsDB"
lst = os.listdir(dirName)
correct=0
detector = MTCNN()
for images in lst:
    imagePath = dirName+"\\"+images
    img = cv2.imread(imagePath)
    faces = detector.detect_faces(img)
    if(len(faces)==0):
        continue
    x1, y1, width, height = faces[0]['box']
    x2, y2 = x1 + width, y1 + height
    imgResized = cv2.resize(img[y1:y2, x1:x2],(227,227))    
    test_image=np.expand_dims(imgResized,axis=0)
    result=AlexNet_loaded_model.predict(test_image,verbose=0)
    if(ResultMap[np.argmax(result)], (images.split("_")[-1]).split(".")[0]):
        correct+=1
    print(correct)
print((correct/len(lst))*100)
