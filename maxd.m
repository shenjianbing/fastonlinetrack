function pnum = maxd(detections)
%% return the maximum of detections per frame
N_frame =length(detections);
pnum =0;
for i = 1:N_frame
    l = detections{i};
    pnum = max(pnum,size(l,1));
end
end
