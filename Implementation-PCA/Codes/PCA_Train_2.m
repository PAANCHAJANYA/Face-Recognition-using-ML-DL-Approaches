n=input('Enter No. of Images for training: ');
L=input('Enter No. of Dominant Eigen Values to keep: ');
M=100; N=90; %Required Image Dimensions
X=zeros(n,(M*N)); %Initialise Data set matrix [X]
T=zeros(n,L); %Initialize Transformed data set [T] in PCA space
for count=1:n
    I=imread(sprintf("C:\\Users\\krish\\Desktop\\Krishna Work\\Face Detection and Recognition\\Implementation - PCA\\TrainDB\\Images_2\\ (%d).JPG",count)); %Reading images
    I=rgb2gray(I);
    I=imresize(I,[M,N]);
    X(count,:)=reshape(I,[1,M*N]); %Reshaping images as 1D vector
end
Xb=X; %Copy database for further use
m=mean(X); %Mean of all Images
for i=1:n
    X(i,:)=X(i,:)-m; %Subtracting Mean from each 1D image
end
Q=(X'*X)/(n-1); %Finding Covariance Matrix
[Evecm, Evalm]=eig(Q); %Getting Eigen values and Eigen vectors of COV matrix [0]
Eval=diag(Evalm); %Extracting all eigen values
[Evalsorted, Index]=sort(Eval,'descend'); %Sorting Eigen Values
Evecsorted=Evecm(:,Index);
Ppca=Evecsorted(:,1:L); %Reduced Transformation matrix [Ppca]
for i=1:n
    T(i,:)=(Xb(i,:)-m)*Ppca; %Projecting each image to PCA space
end