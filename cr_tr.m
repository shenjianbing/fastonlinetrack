function trackRes = cr_tr(objects,framenum,flag)
global scores;
trackRes = [];
for la = 1:size(scores,3);
    for frame = 1:framenum
        l = zeros(1,6);
        l(1) = frame;
        l(2) = la ;
        num = find(scores(1,:,frame)==la);
        if num
            l(5:6) = objects{frame}(num,3:4);
            l(3:4) = objects{frame}(num,1:2) -1/2*objects{frame}(num,3:4);
            trackRes = cat(1,trackRes,l);
        else
            continue;
        end
    end
end
end