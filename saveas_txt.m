function saveas_txt(outDir,sequence_name,trackRes)
% transfer the result to adapt the devkit in MOT2015 https://motchallenge.net

[N,D] = size(trackRes);
T_res = -ones(N,10);
T_res(:,1:6) = trackRes(:,1:6); % frame, id, left, top
% T_res(:,5) = trackRes(:,5)-trackRes(:,3); % width
% T_res(:,6) = trackRes(:,6)-trackRes(:,4); % height
T_res = sortrows(T_res);
dlmwrite([outDir sequence_name '.txt'],T_res,',');
end
