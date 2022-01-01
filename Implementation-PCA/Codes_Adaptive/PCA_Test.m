clc;
camOrTest = input('Do you want an IP Camera to be initiated to test the image (Y/N):', 's');
if camOrTest == 'Y'
    ip = input('Enter the IP Address of the IP Camera Server:', 's');
    cam = ipcam(strcat(strcat('http://', ip), ':8080/video'));
    [workname, workpath] = uigetfile('*.*','Select the workspace to be used for testing');
    load(strcat(workpath, workname));
    timeLimit = input('Enter the time duration (in seconds) for which snapshots should be taken: ');
    fprintf("Get Ready! Snapshots from the IP Camera shall be captured for %d seconds in a few seconds...", timeLimit);
    pause(5);
    totalTime = 0;
    while(totalTime<=timeLimit)
        disp(totalTime);
        tstart = tic;
        img = snapshot(cam);
        originalImg = img;
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
           imgo = insertObjectAnnotation(originalImg, "rectangle", [0,0, 30, 30], 'No face has been detected', 'FontSize',30);
           imshow(imgo);
           totalTime = totalTime + toc(tstart);
           continue;
        end
        img = rgb2gray(img);
        img = imresize(img,[M,N]);
        img = double(reshape(img,[1,M*N]));
        imgpca = (img-m)*Ppca;
        distarray = zeros(n,1);
        for i = 1:n
            distarray(i) = sum(abs(T(i,:)-imgpca));
        end
        [result,indx]=min(distarray);
        if result > threshold
            imgo = insertObjectAnnotation(originalImg, "rectangle", bboxes, 'No face has been matched', 'FontSize',30);
            imshow(imgo);
            totalTime = totalTime + toc(tstart);
            continue;
        else
            recogFile = split(files(indx).name, '_');
            imgo = insertObjectAnnotation(originalImg, "rectangle", bboxes, strcat(recogFile(1),' has been identified'), 'FontSize',30);
            imshow(imgo);
        end
        totalTime = totalTime + toc(tstart);
    end
    clear cam;
    pause(5);
else
    [filename,pathname] = uigetfile('*.*','Select the image to be recognised');
    filewithpath = strcat(pathname,filename);
    img = imread(filewithpath);
    info = imfinfo(filewithpath);
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
    originalImg = img;
    cropped = input('Is the image to be tested already cropped to the face (Y/N): ', 's');
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
           imgo = insertObjectAnnotation(originalImg, "rectangle", [0,0, 30, 30], 'No face has been detected', 'FontSize',30);
           imshow(imgo);
           error('Program terminated');
        end
    end
    [workname, workpath] = uigetfile('*.*','Select the workspace to be used for testing');
    load(strcat(workpath, workname));
    if size(img,3) == 3
        img = rgb2gray(img);
    end
    img = imresize(img,[M,N]);
    imshow(img);
    pause(5);
    img = double(reshape(img,[1,M*N]));
    imgpca = (img-m)*Ppca;
    distarray = zeros(n,1);
    for i = 1:n
        distarray(i) = sum(abs(T(i,:)-imgpca));
    end
    [result,indx]=min(distarray);
    if result > threshold
        imgo = insertObjectAnnotation(originalImg, "rectangle", [0,0, 30, 30], 'No face has been matched', 'FontSize',30);
        imshow(imgo);
    else
        recogFile = split(files(indx).name, '_');
        imgo = insertObjectAnnotation(originalImg, "rectangle", bboxes, strcat(recogFile(1),' has been identified'), 'FontSize',30);
        imshow(imgo);
    end
end