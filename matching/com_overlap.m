function H = com_overlap(pos1,pos2,sz1,sz2)
   h11 = pos1 - 1/2*sz1;
   h12 = pos2 + 1/2*sz1;
   h21 = pos2 - 1/2*sz2;
   h22 = pos2 + 1/2*sz2;
   h1 = cat(2,h11,h12);
   h2 = cat(2,h21,h22);
   H = compute_overlap(h1,h2);
end