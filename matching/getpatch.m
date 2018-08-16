function im_patch = getpatch(now_im,now_pos,now_sz)
      xs = floor(now_pos(1)) + (1:now_sz(1)) - floor(now_sz(1)/2);
      ys = floor(now_pos(2)) + (1:now_sz(2)) - floor(now_sz(2)/2);
      xs(xs < 1) = 1;
      ys(ys < 1) = 1;
      xs(xs > size(now_im,2)) = size(now_im,2);
      ys(ys > size(now_im,1)) = size(now_im,1);
      im_patch =now_im(ys,xs,:);
end