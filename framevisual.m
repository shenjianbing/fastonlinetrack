function framevisual(objects,framenum,im_directory,images)
global scores;
for frame= 1:min(framenum,numel(objects))
    info = objects{frame};
    [m,n] = size(info);
    now_im=imread(fullfile(im_directory,images(frame,1).name));
    now_pos = zeros(m,2);
    now_sz = zeros(m,2);
    for i=1:m
        now_pos(i,:) = floor([info(i,1) info(i,2)]);
        now_sz(i,:) = floor([info(i,3) info(i,4)]);
    end
    imshow(now_im);
    hold on;
    for i =1:m
        if scores(1,i,frame) ~= 0
            rect_position =[now_pos(i,:) - now_sz(i,:)/2, now_sz(i,:)];
            rectangle('Position',rect_position, 'EdgeColor','r','LineWidth',1.5);
            fx = now_pos(i,1);
            fy = now_pos(i,2)- now_sz(i,2)/2- 10;
            text_handle = text(fx,fy,int2str(scores(1,i,frame)));
            set(text_handle, 'color', [1 0 0]);
        end
    end
    drawnow;
    hold off;
end