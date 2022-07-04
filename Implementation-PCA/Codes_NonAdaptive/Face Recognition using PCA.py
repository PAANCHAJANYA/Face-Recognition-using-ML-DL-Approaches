import numpy as np
import os
import cv2
from mtcnn.mtcnn import MTCNN
from PIL import Image
from numpy import asarray
from numpy.linalg import eig
from numpy import linalg as npla

# TRAINING SEGEMENT

trainDirectory = "C:\\Users\\krish\\Desktop\\Krishna Work\\Face Detection and Recognition\\Face Datasets\\TrainDB\\FriendsDB\\"
imageExt = input('Enter the Image type (JPG/JPEG/PNG/GIF/PGM):')
cropped = input('Are the images to be trained already cropped to the face (Y/N): ')
n = 0
L = int(input('Enter the number of dominant eigen Values to be considered (max:-9000): '))
M = 100
N = 90
X = np.zeros((len(os.listdir(trainDirectory)),M * N))
T = np.zeros((len(os.listdir(trainDirectory)),L))
mtcnn_detector = MTCNN()
files = []
for images in os.listdir(trainDirectory):
    if(images.endswith('.'+imageExt.lower())):
        files.append(images)
        filePath = trainDirectory + images
        img = cv2.imread(filePath)
        if cropped == 'N':
            results = mtcnn_detector.detect_faces(img)
            if(len(results)==0):
                continue
            x,y,w,h = results[0]['box']
            x, y = abs(x), abs(y)
            image = Image.fromarray(img[y:y+h, x:x+w])
            image = image.resize((M,N))
            face_array = asarray(image)
            gray = cv2.cvtColor(face_array, cv2.COLOR_BGR2GRAY)
            cv2.imshow("A", gray)
            X[n,:] = gray.reshape(1,M*N)
            print(X[n,:])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            image = Image.fromarray(img)
            image = image.resize((M,N))
            face_array = asarray(image)
            gray = cv2.cvtColor(face_array, cv2.COLOR_BGR2GRAY)
            X[n,:] = gray.reshape(1,M*N)
        n = n + 1
cv2.destroyAllWindows()
Xb = X
m = X.mean(axis=0)
print(m.shape)
X = X - m
Q = np.dot(np.transpose(X),X)/(n - 1)
Evalm , Evecm = eig(Q)
idx = np.argsort(Evalm)
Evalm = Evalm[idx]
Evecm = Evecm[:,idx]
print("Eigen vectors:", Evecm.shape)
Ppca = Evecm[:,M*N-L:M*N]
for i in range(n):
    T[i,:] = np.dot((Xb[i,:] - m),Ppca)
print(T.shape)
minEdList = np.zeros((1, n))
for i in range(n):
    minEd = 99999999999999999999;
    for j in range(n):
        if j == i:
            continue
        if minEd > np.sum(np.abs(T[i,:]-(np.dot((Xb[j,:]-m),Ppca)))):
            minEd = np.sum(np.abs(T[i,:]-(np.dot((Xb[j,:]-m),Ppca))))
    minEdList[0, i] = minEd
threshold = 0.8*np.max(minEdList)
print(threshold)

# TESTING SEGEMENT
# Uses threshold, Ppca, T, M, N, m, n, files variables assigned in TRAINING SEGEMENT


vs = cv2.VideoCapture(0)
while(vs.isOpened()):
    ok, frame = vs.read()
    if ok:
        results = mtcnn_detector.detect_faces(frame)
        if(len(results)==0):
            continue
        x,y,w,h = results[0]['box']
        x, y = abs(x), abs(y)
        image = Image.fromarray(frame[y:y+h, x:x+w])
        image = image.resize((M,N))
        face_array = asarray(image)
        gray = cv2.cvtColor(face_array, cv2.COLOR_BGR2GRAY)
        grayreshaped = gray.reshape(1,M*N)
        imgpca = np.dot((grayreshaped - m), Ppca)
        distarray = np.zeros((n,1))
        for i in range(n):
            distarray[i] = np.sum(np.abs(T[i,:]-imgpca))
        result = np.min(distarray)
        indx = np.argmin(distarray)
        if result > threshold:
            cv2.putText(frame, "No Match", (10, 70), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 1)
        else:
            recogFile = files[indx].split('_')
            cv2.putText(frame, recogFile[0], (10, 70), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 1)
        cv2.imshow("Video LiveFeed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
vs.release()
cv2.destroyAllWindows()
