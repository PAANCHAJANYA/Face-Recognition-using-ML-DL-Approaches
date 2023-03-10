import numpy as np
from tensorflow.python.keras.models import load_model
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Normalizer, LabelEncoder
from sklearn.svm import SVC
import os, pickle
from Preprocess import extract_face, load_face, load_dataset
from keras_facenet import FaceNet

trainX, trainy = load_dataset('./faces-dataset/train/')
print(trainX.shape, trainy.shape)

testX, testy = load_dataset('./faces-dataset/val/')
print(testX.shape, testy.shape)

embedder = FaceNet()

emdTrainX = embedder.embeddings(trainX)
print(emdTrainX.shape)

emdTestX = embedder.embeddings(testX)
print(emdTestX.shape)

print("Dataset: Train=%d, Test=%d" % (emdTrainX.shape[0], emdTestX.shape[0]))

in_encoder = Normalizer()
emdTrainX_norm = in_encoder.transform(emdTrainX)
emdTestX_norm = in_encoder.transform(emdTestX)

out_encoder = LabelEncoder()
out_encoder.fit(trainy)

np.save('classes.npy', out_encoder.classes_)

trainy_enc = out_encoder.transform(trainy)
testy_enc = out_encoder.transform(testy)

model = SVC(kernel='linear', probability=True)
model.fit(emdTrainX_norm, trainy_enc)

yhat_train = model.predict(emdTrainX_norm)
yhat_test = model.predict(emdTestX_norm)

score_train = accuracy_score(trainy_enc, yhat_train)
score_test = accuracy_score(testy_enc, yhat_test)

print('Accuracy: Train=%.3f, Test=%.3f' % (score_train*100, score_test*100))

with open('SVCtrainedModel.pkl', 'wb') as f:
    pickle.dump(model, f)
