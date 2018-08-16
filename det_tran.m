function detections = det_tran(detections)
  num = max(detections(:,1));
  flag = 1;
  det = cell(1,num);
  for flag= 1:num   
      det{flag} = detections(find(detections(:,1) == flag),:);
      det{flag}(:,8:10) = [];
      det{flag}(:,1:2) = [];
      det{flag}(:,3:4) = det{flag}(:,1:2)+det{flag}(:,3:4);
      det{flag} = floor(det{flag});
      det{flag}(:,5) = det{flag}(:,5)/100;
      det{flag}(find(det{flag}(:,5)<0.1),:)= [];  
  end
  detections = det;
end