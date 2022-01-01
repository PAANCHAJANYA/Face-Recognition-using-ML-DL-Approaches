trainDirectory = uigetdir(title='Select the folder whose content needs to be trained');
trainDirectory = strcat(trainDirectory, "\\");
imageExt = input('Enter the Image type (JPG/JPEG/PNG/GIF/PGM):', 's');
files = dir(fullfile(trainDirectory, strcat('*', imageExt)));
n = length(files);
cropped = input('Are the images to be trained already cropped to the face (Y/N): ', 's');
L = input('Enter the number of dominant eigen Values to be considered (max:-9000): ');
M = 100; N = 90;
X = zeros(n,(M*N));
T = zeros(n,L);
for count = 1:n
    filePath = strcat(trainDirectory, files(count).name);
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
    X(count,:) = reshape(img,[1,M*N]);
end
Xb = X;
m = mean(X);
for i = 1:n
    X(i,:) = X(i,:) - m;
end
Q = (X'*X)/(n-1);
[Evecm, Evalm] = eig(Q);
Eval = diag(Evalm);
[Evalsorted, Index] = sort(Eval, 'descend');
Evecsorted = Evecm(:, Index);
Ppca = Evecsorted(:, 1:L);
for i = 1:n
    T(i,:) = (Xb(i,:)-m)*Ppca;
end
if n <= 300
    minEdList = zeros(1, n);
    for i = 1:n
        minEd = 99999999999999999999;
        for j = 1:n
            if j == i
                continue;
            end
            if minEd > sum(abs(T(i,:)-((Xb(j,:)-m)*Ppca)))
                minEd = sum(abs(T(i,:)-((Xb(j,:)-m)*Ppca)));
            end
        end
        minEdList(1, i) = minEd;
    end
elseif n <= 1000
    minEdList = zeros(1, ceil(n/5));
    for i = 1:5:n
        minEd = 99999999999999999999;
        for j = 1:n
            if j == i
                continue;
            end
            if minEd > sum(abs(T(i,:)-((Xb(j,:)-m)*Ppca)))
                minEd = sum(abs(T(i,:)-((Xb(j,:)-m)*Ppca)));
            end
        end
        minEdList(1, ceil(i/5)) = minEd;
    end
elseif n <= 5000
    minEdList = zeros(1, ceil(n/25));
    for i = 1:25:n
        minEd = 99999999999999999999;
        for j = 1:n
            if j == i
                continue;
            end
            if minEd > sum(abs(T(i,:)-((Xb(j,:)-m)*Ppca)))
                minEd = sum(abs(T(i,:)-((Xb(j,:)-m)*Ppca)));
            end
        end
        minEdList(1, ceil(i/25)) = minEd;
    end
elseif n <= 10000
    minEdList = zeros(1, ceil(n/50));
    for i = 1:50:n
        minEd = 99999999999999999999;
        for j = 1:n
            if j == i
                continue;
            end
            if minEd > sum(abs(T(i,:)-((Xb(j,:)-m)*Ppca)))
                minEd = sum(abs(T(i,:)-((Xb(j,:)-m)*Ppca)));
            end
        end
        minEdList(1, ceil(i/50)) = minEd;
    end
elseif n <= 15000
    minEdList = zeros(1, ceil(n/75));
    for i = 1:75:n
        minEd = 99999999999999999999;
        for j = 1:n
            if j == i
                continue;
            end
            if minEd > sum(abs(T(i,:)-((Xb(j,:)-m)*Ppca)))
                minEd = sum(abs(T(i,:)-((Xb(j,:)-m)*Ppca)));
            end
        end
        minEdList(1, ceil(i/75)) = minEd;
    end
else
    minEdList = zeros(1, ceil(n/100));
    for i = 1:100:n
        minEd = 99999999999999999999;
        for j = 1:n
            if j == i
                continue;
            end
            if minEd > sum(abs(T(i,:)-((Xb(j,:)-m)*Ppca)))
                minEd = sum(abs(T(i,:)-((Xb(j,:)-m)*Ppca)));
            end
        end
        minEdList(1, ceil(i/100)) = minEd;
    end
end
clear max;
threshold = 0.8*max(minEdList);