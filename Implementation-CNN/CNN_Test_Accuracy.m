testDirectory = uigetdir(title='Select the folder whose content needs to be tested');
testDirectory = strcat(testDirectory, "\\");
imageExt = input('Enter the Image type (JPG/JPEG/PNG/GIF/PGM):', 's');
cropped = input('Are the images in this test folder already cropped to the face (Y/N): ', 's');
testFiles = dir(fullfile(testDirectory, strcat('*', imageExt)));
nTest = length(testFiles);
[workname, workpath] = uigetfile('*.*','Select the workspace to be used for testing');
load(strcat(workpath, workname));
correct=0;
for count=1:nTest
    filePath = strcat(testDirectory, testFiles(count).name);
    imgOrg = imread(filePath);
    info = imfinfo(filePath);
    hasField = isfield(info, 'Orientation');
    if hasField
        if info.Orientation == 2
            imgOrg = flip(imgOrg, 2);
        elseif info.Orientation == 3
            imgOrg = imrotate(imgOrg, 180);
        elseif info.Orientation == 4
            imgOrg = imrotate(imgOrg, 180);
            imgOrg = flip(imgOrg, 2);
        elseif info.Orientation == 5
            imgOrg = imrotate(imgOrg, -90);
            imgOrg = flip(imgOrg, 2);
        elseif info.Orientation == 6
            imgOrg = imrotate(imgOrg, -90);
        elseif info.Orientation == 7
            imgOrg = imrotate(imgOrg, 90);
            imgOrg = flip(imgOrg, 2);
        elseif info.Orientation == 8
            imgOrg = imrotate(imgOrg, 90);
        end
    end
    if(size(imgOrg,3)==1)
        imgEdited = zeros(size(imgOrg,1), size(imgOrg,2), 3);
        imgEdited(:,:,1) = imgOrg;
        imgEdited(:,:,2) = imgOrg;
        imgEdited(:,:,3) = imgOrg;
        img = imgEdited/255;
    else
        img = imgOrg;
    end
    if cropped == 'N'
        [bboxes, scores, landmarks] = mtcnn.detectFaces(img);
        if ~isempty(bboxes)
            if(size(bboxes, 1)>1)
                img = imcrop(img, bboxes(1,:));
            else
                img = imcrop(img, bboxes);
            end
        else
            continue;
        end
    end
    img = imresize(img,[227,227]);
    imshow(img);
    [predict,score] = classify(newnet,img);
    if(max(score)>0.5)
        recogFile = split(testFiles(count).name, '_');
        if(strcmp(recogFile{1},string(predict))==1)
            correct = correct+1;
        end
    end
end
fprintf("Accuracy of CNN in recognition: %f\n", (correct/nTest)*100);