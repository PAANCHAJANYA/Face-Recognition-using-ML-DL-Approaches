#!/usr/bin/env python
# coding: utf-8

# In[16]:


from PIL import Image
import glob
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt 
import cv2
import math
from mtcnn.mtcnn import MTCNN


# In[34]:


cropped = False
color = True
train_dir = 'C:\\Users\\krish\\Desktop\\Krishna Work\\Face Detection and Recognition\\Face Datasets\\TrainDB\\FriendsDB\\*.jpg'
test_dir = 'C:\\Users\\krish\\Desktop\\Krishna Work\\Face Detection and Recognition\\Face Datasets\\TestDB\\FriendsDB\\*.jpg'
L = 250


# In[18]:


#Training

mtcnn_detector = MTCNN()
image=[]
imagename = []
flattened_images = []

def griddisplay(image_list):
    fig1, axes_array = plt.subplots(math.ceil(len(image_list)/10), 10)
    fig1.set_size_inches(10,math.ceil(len(image_list)/10))
    k=0
    for row in range(math.ceil(len(image_list)/10)):
        for col in range(10):
            if k>=len(image_list):
                break
            image_plot = axes_array[row][col].imshow(image_list[k],cmap=plt.cm.gray) 
            axes_array[row][col].axis('off')
            k = k+1
    plt.show()

for filename in glob.glob(train_dir):
    print(filename)
    im = Image.open(filename)
    im1=im.convert('RGB')
    img = np.asarray(im1)
    if not cropped:
        results = mtcnn_detector.detect_faces(img)
        if(len(results)==0):
            continue
        x,y,w,h = results[0]['box']
        x, y = abs(x), abs(y)
        if color:
            im2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            im2 = img
        im = np.asarray(im2,dtype=float)/255.0
        image.append(cv2.resize(im[y:y+h, x:x+w],(128,128),interpolation=cv2.INTER_LINEAR))
    else:
        if color:
            im2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            im2 = img
        im = np.asarray(im2,dtype=float)/255.0
        image.append(cv2.resize(im,(128,128),interpolation=cv2.INTER_LINEAR))
    imagename.append(filename.split("\\")[-1])

print('Original Images')
griddisplay(image)
for i in range(len(image)):
    flattened_images.append(image[i].flatten())
A = np.matrix(flattened_images)

mean = np.mean(A,0)
resized_mean = mean.reshape(128,128)
plt.imshow(resized_mean,cmap=plt.cm.gray)
plt.axis('off')
plt.title('Mean Face')
plt.show()

zero_mean = []
row = 0
Zero_mean_matrix = np.ones((len(image),16384))
for values in flattened_images:
    zm = A[row,:] - mean
    Zero_mean_matrix[row,:] = zm
    zero_mean.append(zm.reshape(128,128).tolist())
    row = row + 1
print('Zero mean faces')
griddisplay(zero_mean)

d = (np.dot(Zero_mean_matrix,np.transpose(Zero_mean_matrix)))/128
u_list =[]
w2, v2 = la.eigh(d)

for ev in v2:
    ev_transpose = np.transpose(np.matrix(ev))
    u = np.dot(np.transpose(Zero_mean_matrix),ev_transpose)
    u = u / np.linalg.norm(u)
    u_i= u.reshape(128,128)
    u_list.append(u_i)

print('Eigen faces')
griddisplay(u_list)

weights = np.zeros((len(image),L))
matrixU = np.zeros((16384,L))
c = 0
for val in range(L-1,-1,-1):
    matrixU[:,c] = u_list[val].flatten()
    c = c+1
for face_num in range(len(image)):
    weights[face_num,:] = np.dot(Zero_mean_matrix[face_num,:],matrixU)


# In[35]:


np.savez('PCA_Data.npz', MEAN=mean, U=matrixU, WEIGHTS=weights, NAMES=imagename)

data = np.load('PCA_Data.npz')
mean = data['MEAN']
matrixU = data['U']
weights = data['WEIGHTS']
imagename = data['NAMES']


# In[36]:


#Testing

count = 0
threshold = 28000
correct = 0
for filename in glob.glob(test_dir):
    im = Image.open(filename)
    img = np.asarray(im)
    if not cropped:
        results = mtcnn_detector.detect_faces(img)
        if(len(results)==0):
            continue
        x,y,w,h = results[0]['box']
        x, y = abs(x), abs(y)
        if color:
            im2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            im2 = img
        im = np.asarray(im2,dtype=float)/255.0
        img = cv2.resize(im[y:y+h, x:x+w],(128,128),interpolation=cv2.INTER_LINEAR).flatten()
    else:
        if color:
            im2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            im2 = img
        im = np.asarray(im2,dtype=float)/255.0
        img = cv2.resize(im,(128,128),interpolation=cv2.INTER_LINEAR).flatten()
    print((img-mean).shape)
    imgpca = (img - mean).dot(matrixU);
    distarray = np.zeros((len(image),1));
    for i in range(len(image)):
        distarray[i] = np.sum(np.abs(weights[i,:]-imgpca));
    [result,indx]=[np.min(distarray),np.argmin(distarray)];
    print(result, imagename[indx].split("_")[0], filename.split("\\")[-1].split("_")[0])
    if filename.split("\\")[-1].split("_")[0]==imagename[indx].split("_")[0]:
        correct+=1
    count+=1
    print(count, correct)
print(correct)
print((correct/count)*100)

