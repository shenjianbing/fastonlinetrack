function object=info_get(x)
%% change detection format
[m,n] =size(x);
if m==1
    object=x;
    for num=1:n
        detectionresult = x{1,num};
        [m1,n1] = size(detectionresult);
        if  n1==5
            tempresult = zeros(m1,4);
            tempresult(:,1) = floor(1/2*(detectionresult(:,1)+detectionresult(:,3)));
            tempresult(:,2) = floor(1/2*(detectionresult(:,2)+detectionresult(:,4)));
            tempresult(:,3) = detectionresult(:,3) - detectionresult(:,1);
            tempresult(:,4) = detectionresult(:,4) - detectionresult(:,2);
        else
            error('wrong detection format');
        end
        object{num}=tempresult;
    end
else
    error('wrong detection format');
end
end