from Preprocess import extract_face
from tensorflow.python.keras.models import load_model
from sklearn.preprocessing import Normalizer, LabelEncoder
import pickle
import numpy as np
from keras_facenet import FaceNet

in_encoder = Normalizer()
out_encoder = LabelEncoder()
out_encoder.classes_ = np.load('classes.npy')

with open('SVCtrainedModel.pkl', 'rb') as f:
    model = pickle.load(f)

random_face = extract_face("C:\\Users\\krish\\Desktop\\Mitra Vision\\static\\KrishnaPaanchajanya.jpg")
embedder = FaceNet()
random_face_emd = in_encoder.transform(embedder.embeddings([random_face]))[0]

samples = np.expand_dims(random_face_emd, axis = 0)
yhat_class = model.predict(samples)
yhat_prob = model.predict_proba(samples)

class_index = yhat_class[0]
class_probability = yhat_prob[0, class_index] * 100
predicted_name = out_encoder.inverse_transform(yhat_class)[0]
all_names = out_encoder.inverse_transform([i for i in range(len(out_encoder.classes_))])

print("Predicted Probabilities: ")
for i, name in enumerate(all_names):
    print(name, ": ", yhat_prob[0][i] * 100)
print('Predicted: %s' % predicted_name)
