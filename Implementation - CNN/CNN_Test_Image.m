[filename,pathname] = uigetfile('*.*','Select the image to be recognised');
filewithpath = strcat(pathname,filename);
[workname, workpath] = uigetfile('*.*','Select the workspace to be used for testing');
load(strcat(workpath, workname));
the_Image = imread(filewithpath);
originalImg = the_Image;
if(size(the_Image,3)==1)
    imgEdited = zeros(size(the_Image,1), size(the_Image,2), 3);
    imgEdited(:,:,1) = the_Image;
    imgEdited(:,:,2) = the_Image;
    imgEdited(:,:,3) = the_Image;
    the_Image = imgEdited/255;
end
[bboxes, scores, landmarks] = mtcnn.detectFaces(the_Image);
if ~isempty(bboxes)
    if(size(bboxes, 1)>1)
        for itr=1:size(bboxes, 1)
            img = imcrop(the_Image, bboxes(itr,:));
            img = imresize(img,[227,227]);
            [predict,score] = classify(newnet,img);
            if(max(score)>0.5)
                disp(predict);
            else
                disp("None of the persons trained is recognised!");
            end
        end
    else
        img = imcrop(the_Image, bboxes);
        img = imresize(img,[227,227]);
        [predict,score] = classify(newnet,img);
        if(max(score)>0.5)
            disp(predict);
        else
            disp("None of the persons trained is recognised!");
        end
    end
else
    disp("No face has been detected!");
end