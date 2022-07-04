[filename,pathname] = uigetfile('*.*','Select the video to be recognised');
filewithpath = strcat(pathname,filename);
[workname, workpath] = uigetfile('*.*','Select the workspace to be used for testing');
load(strcat(workpath, workname));
the_Video = VideoReader(filewithpath);
totalFrames = the_Video.NumFrames;
for frameItr = 10:10:totalFrames
    video_Frame = read(the_Video, frameItr);
    originalImg = video_Frame;
    [bboxes, scores, landmarks] = mtcnn.detectFaces(video_Frame);
    if ~isempty(bboxes)
        if(size(bboxes, 1)>1)
            for itr=1:size(bboxes, 1)
                img = imcrop(video_Frame, bboxes(itr,:));
                img = imresize(img,[227,227]);
                [predict,score] = classify(newnet,img);
                if(max(score)>0.9)
                    originalImg = insertObjectAnnotation(originalImg, "rectangle", bboxes(itr,:), string(predict));
                else
                    originalImg = insertObjectAnnotation(originalImg, "rectangle", bboxes(itr,:), "NA");
                end
            end
            imshow(originalImg);
        else
            img = imcrop(video_Frame, bboxes);
            img = imresize(img,[227,227]);
            [predict,score] = classify(newnet,img);
            if(max(score)>0.9)
                originalImg = insertObjectAnnotation(originalImg, "rectangle", bboxes, predict);
            else
                originalImg = insertObjectAnnotation(originalImg, "rectangle", bboxes, "NA");
            end
            imshow(originalImg);
        end
    else
        continue;
    end
end