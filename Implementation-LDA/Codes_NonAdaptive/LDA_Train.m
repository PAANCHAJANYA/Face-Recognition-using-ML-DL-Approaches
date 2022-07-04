trainDirectory = uigetdir(title='Select the folder whose content needs to be trained');
trainDirectory = strcat(trainDirectory, "\\");
imageExt = input('Enter the Image type (JPG/JPEG/PNG/GIF/PGM):', 's');
cropped = input('Are the images to be trained already cropped to the face (Y/N): ', 's');
L = input('Enter the number of dominant eigen Values to be considered (max:-9000): ');
files = dir(fullfile(trainDirectory, strcat('*', imageExt)));
filesTable = struct2table(files);
sortedFilesTable = sortrows(filesTable, 'name');
files = table2struct(sortedFilesTable);
n = length(files);
subjectName = strings([n, 1]);
for i=1:n
    tempSubjectName = split(files(i).name, '_');
    subjectName(i) = char(tempSubjectName(1));
end
c = categorical(subjectName);
imagePerSubject = countcats(c);
subjects = length(imagePerSubject);
M = 100; N = 90;
X = zeros(n,(M*N));
Xb = zeros(n,(M*N));
tempX = zeros(max(imagePerSubject), (M*N));
mClass = zeros(subjects, (M*N));
SSubjects = zeros(M*N, M*N, subjects);
Sw = zeros(M*N, M*N);
Sb = zeros(M*N, M*N);
T = zeros(n,L);
for count1 = 1:subjects
    clear max;
    tempX = zeros(max(imagePerSubject), (M*N));
    for count2 = 1+sum(imagePerSubject(1:count1-1)):sum(imagePerSubject(1:count1))
        filePath = strcat(trainDirectory, files(count2).name);
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
            %faceDetector = vision.CascadeObjectDetector('ClassificationModel', 'FrontalFaceCART');
            %faceDetector.MergeThreshold = 4;
            %bboxes = faceDetector(img);
            [bboxes, scores, landmarks] = mtcnn.detectFaces(img);
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
        tempIndex = count2-sum(imagePerSubject(1:count1-1));
        tempX(tempIndex,:) = reshape(img, [1, M*N]);
        X(count2,:) = reshape(img,[1,M*N]);
        Xb(count2,:) = reshape(img,[1,M*N]);
    end
    mClass(count1,:)= mean(tempX);
    for count2 = 1+sum(imagePerSubject(1:count1-1)):sum(imagePerSubject(1:count1))
        X(count2,:) = X(count2,:) - mClass(count1,:);
        SSubjects(:, :, count1)=SSubjects(:, :, count1)+(X(count2, :))'*(X(count2,:));
    end
end
m = mean(Xb);
for i=1:subjects
    Sw = Sw + SSubjects(:, :, i);
    Sb = Sb + imagePerSubject(i)*(mClass(i,:)-m)'*(mClass(i,:)-m);
end
Q = pinv(Sw)*Sb;
[Evecm, Evalm] = eig(Q);
Eval = diag(Evalm);
[Evalsorted, Index] = sort(Eval, 'descend');
Evecsorted = Evecm(:, Index);
Plda = Evecsorted(:, 1:L);
for i = 1:n
    T(i,:) = (Xb(i,:)-m)*Plda;
end
if n <= 300
    minEdList = zeros(1, n);
    for i = 1:n
        minEd = 99999999999999999999;
        for j = 1:n
            if j == i
                continue;
            end
            if minEd > sum(abs(T(i,:)-((Xb(j,:)-m)*Plda)))
                minEd = sum(abs(T(i,:)-((Xb(j,:)-m)*Plda)));
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
            if minEd > sum(abs(T(i,:)-((Xb(j,:)-m)*Plda)))
                minEd = sum(abs(T(i,:)-((Xb(j,:)-m)*Plda)));
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
            if minEd > sum(abs(T(i,:)-((Xb(j,:)-m)*Plda)))
                minEd = sum(abs(T(i,:)-((Xb(j,:)-m)*Plda)));
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
            if minEd > sum(abs(T(i,:)-((Xb(j,:)-m)*Plda)))
                minEd = sum(abs(T(i,:)-((Xb(j,:)-m)*Plda)));
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
            if minEd > sum(abs(T(i,:)-((Xb(j,:)-m)*Plda)))
                minEd = sum(abs(T(i,:)-((Xb(j,:)-m)*Plda)));
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
            if minEd > sum(abs(T(i,:)-((Xb(j,:)-m)*Plda)))
                minEd = sum(abs(T(i,:)-((Xb(j,:)-m)*Plda)));
            end
        end
        minEdList(1, ceil(i/100)) = minEd;
    end
end
clear max;
threshold = 0.8*max(minEdList);