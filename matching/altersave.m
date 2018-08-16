function alternative = altersave(now_im,frame,pre_im,pre_pos,pre_sz,t1,label,virflag,non_compressed_features, compressed_features, w2c)
    alternative = zeros(1,8);
    alternative(1,1:2) = pre_pos;
    alternative(1,3:4) = pre_sz;
    alternative(1,5) = frame -1;
    alternative(1,6) = t1;
    alternative(1,7) = label;
    alternative(1,8) = virflag + 1;
end