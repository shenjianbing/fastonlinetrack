function detections = ndet_tran(detections,num)
%% divide the detection matrix into cells according to the frame number
global params;
for flag= 1:num
    det{flag} = detections(detections(:,1) == flag,:);
    det{flag}(:,1:2) = [];
    
    % if your detection format is [x1,y1,x2,y2] not [x,y,w,h], comment the
    % line below
    det{flag}(:,3:4) = det{flag}(:,1:2)+det{flag}(:,3:4);
    
    det{flag}(:,1:4) = floor(det{flag}(:,1:4));
    det{flag}(:,6:end)=[];
end
detections = det;
end