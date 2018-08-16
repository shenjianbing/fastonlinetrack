function scores_resize_myversion(objects)
%% change score to the same format as object
global scores;
for i=1:size(scores,3)
    max=size(objects{i},1)+1; 
    scores(1,max:size(scores,2),i)=0; 
end
end

