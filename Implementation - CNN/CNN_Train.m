trainDirectory = uigetdir(title='Select the folder whose content needs to be trained');
trainDirectory = strcat(trainDirectory, "\");
cropped = input('Are the images to be trained already cropped to the face (Y/N): ', 's');
imageExt = input('Enter the Image type (JPG/JPEG/PNG/GIF/PGM):', 's');
subjects = length(dir(trainDirectory))-2;
files = dir(fullfile(trainDirectory, '**', strcat('*', imageExt)));
n = length(files);
copyfile(trainDirectory, "temp");
filesToDelete = dir(fullfile("temp", '**', strcat('*', imageExt)));
for count1 = 1:n
    deleteFilePerson = split(filesToDelete(count1).name, '_');
    deleteFilePath = strcat(strcat(strcat("temp\", deleteFilePerson(1)),"\"), filesToDelete(count1).name);
    delete(deleteFilePath);
end
for count2 = 1:n
    currentFilePerson = split(files(count2).name, '_');
    filePath = strcat(strcat(strcat(trainDirectory, currentFilePerson(1)),"\"), files(count2).name);
    imgOrg = imread(filePath);
    if(size(imgOrg,3)==1)
        imgEdited = zeros(size(imgOrg,1), size(imgOrg,2), 3);
        imgEdited(:,:,1) = imgOrg;
        imgEdited(:,:,2) = imgOrg;
        imgEdited(:,:,3) = imgOrg;
        imgOrg = imgEdited/255;
    end
    if(cropped=='N')
        [bboxes, score, landmarks] = mtcnn.detectFaces(imgOrg);
        if ~isempty(bboxes)
            if(size(bboxes, 1)>1)
                img = imcrop(imgOrg, bboxes(1,:));
                img = imresize(img,[227,227]);
            else
                img = imcrop(imgOrg, bboxes);
                img = imresize(img,[227,227]);
            end
        else
            continue;
        end
    else
        img = imresize(imgOrg,[227,227]);
    end
    currentFileName = split(files(count2).name, '.');
    imwrite(img, strcat(strcat(strcat("temp\", currentFilePerson(1)),"\"),strcat(currentFileName(1),".jpg")));
end
dataset = imageDatastore("temp",'IncludeSubfolders',true,'LabelSource','foldernames');
dataset.ReadFcn = @(loc)imresize(imread(loc),[227,227]);
[Train ,Test] = splitEachLabel(dataset,0.8,'randomized');
fc = fullyConnectedLayer(subjects);
net = alexnet;
ly = net.Layers;
ly(23) = fc;
cl = classificationLayer;
ly(25) = cl;
learning_rate = 0.00001;
opts = trainingOptions("rmsprop","InitialLearnRate",learning_rate,'MaxEpochs',20,'MiniBatchSize',64,'Plots','training-progress');
[newnet,info] = trainNetwork(Train, ly, opts);
[predict,scores] = classify(newnet,Test);
names = Test.Labels;
pred = (predict==names);
s = size(pred);
acc = sum(pred)/s(1);
fprintf('The accuracy of the test set is %f %% \n',acc*100);
status = rmdir("temp", 's');