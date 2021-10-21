clc;
load pcadb_2; %Loading pcadb.mat file
%pcadb loads followings in workspace
%Image size [M=100,N=90]
%Mean Image [m]
%Reduced Eigen vectors transformation matrix [Ppca]
%Transformed dataset matrix[I]
[filename,pathname]=uigetfile('*.*','Select the Input Image');
filewithpath=strcat(pathname,filename);
img=imread(filewithpath);
imgo=img;
img=rgb2gray(img);
img=imresize(img,[M,N]);
img=double(reshape(img,[1,M*N]));
imgpca=(img-m)*Ppca; %Projecting query image to PCA space
distarray=zeros(n,1); %initialize difference array
for i=1:n
    distarray(i)=sum(abs(T(i,:)-imgpca)); %Finding L1 distance
end
[result,indx]=min(distarray); %Getting best match
resultimg=imread(sprintf("C:\\Users\\krish\\Desktop\\Krishna Work\\Face Detection and Recognition\\Implementation - PCA\\TrainDB\\Images_2\\ (%d).JPG",indx)); %Getting best matched image
%Plotting Images
subplot(121)
imshow(imgo);
title('Query Face');
subplot(122)
imshow(resultimg);
title('Recognized Face');