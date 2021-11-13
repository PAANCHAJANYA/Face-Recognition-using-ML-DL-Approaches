testFiles = dir(fullfile(testDirectory, strcat('*', imageExt)));
nTest = length(testFiles);
correct = zeros(1, 15);
for count=1:nTest
    filePath = strcat(testDirectory, testFiles(count).name);
    img = imread(filePath);
    info = imfinfo(filePath);
    hasField = isfield(info, 'Orientation');
    if hasField
        if info.Orientation == 2
            img = flip(img, 2);
        elseif info.Orientation == 3
            img = imrotate(img, 180);
        elseif info.Orientation == 4
            img = imrotate(img, 180);
            img = flip(img, 2);
        elseif info.Orientation == 5
            img = imrotate(img, -90);
            img = flip(img, 2);
        elseif info.Orientation == 6
            img = imrotate(img, -90);
        elseif info.Orientation == 7
            img = imrotate(img, 90);
            img = flip(img, 2);
        elseif info.Orientation == 8
            img = imrotate(img, 90);
        end
    end
    if cropped == 'N'
        faceDetector = vision.CascadeObjectDetector('ClassificationModel', 'FrontalFaceCART');
        faceDetector.MergeThreshold = 4;
        bboxes = faceDetector(img);
        if ~isempty(bboxes)
            if(size(bboxes, 1)>1)
                max = bboxes(1, 3)*bboxes(1, 4);
                maxRow = 1;
                for itr = 2:size(bboxes, 1)
                    if max < bboxes(itr, 3)*bboxes(itr, 4)
                        max = bboxes(itr, 3)*bboxes(itr, 4);
                        maxRow = itr;
                    end
                end
                bboxes = bboxes(maxRow, :);
            end
            img = imcrop(img, bboxes);
        else
            continue;
        end
    end
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    img = imresize(img,[M,N]);
    img = double(reshape(img,[1,M*N]));
    imgpca = (img-m)*Ppca;
    distarray = zeros(n,1);
    for i = 1:n
        distarray(i) = sum(abs(T(i,:)-imgpca));
    end
    [result,indx]=min(distarray);
    for thresholdItr=1:15
        if result > thresholdPossibilities(1, thresholdItr)
            continue;
        else
            recogFile = split(files(indx).name, '_');
            actualFile = split(testFiles(count).name, '_');
            if(recogFile{1}==actualFile{1})
                correct(1, thresholdItr)=correct(1, thresholdItr)+1;
            end
        end
    end
end
clear max;
[max, thresholdIndex] = max(correct);
threshold = thresholdPossibilities(thresholdIndex);