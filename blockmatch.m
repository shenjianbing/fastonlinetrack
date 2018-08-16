function [detections,objects,labels,flag,scores,resulttmp,results,virscore,virobjects,virresults] =blockmatch(detections,objects,framenum,im_directory,images,params,pnum)
%% the main function matching objects in adjacent frames 
%% initialize
object_pnum = ceil((pnum+6)/5)*5;
scores = zeros(object_pnum,object_pnum,numel(images));
results = zeros(1,object_pnum,numel(images));
resulttmp = zeros(1,object_pnum,numel(images));
virobjects = {};
virresults =cell(1,framenum);
virscore = cell(1,framenum);
labels = zeros(1,object_pnum,framenum);
flag = 1;
virflag =0;
non_compressed_features = {'gray'};
compressed_features = {'cn'};
temp = load('w2crs');
w2c = temp.w2crs;
for frame= 1:min(framenum,numel(objects))
    %% visualize process
    clc;
    fprintf('Processing %.1f%%\n',frame*100/min(framenum,numel(objects)));
    %% preprocessing
    infor = objects{frame};
    m = size(infor,1);  
    if frame == 1
        pre_m = m;
        for i=1:m
            scores(i,i,frame) =  1;
            results(1,i,frame) = i;
            labels(1,i,frame) = flag;
            flag = flag + 1;
        end
    else
        % save previous information
        pre_im= now_im;
        pre_pos = now_pos;
        pre_sz= now_sz;
    end    
    %% add virtual objects
    vm = size(virobjects,1);
    if vm
        fr = frame - virobjects(:,5);
        outd =  find(fr > 5);
        virobjects(outd,:) = [];
    end
    vm = size(virobjects,1);
    %% load current information
    now_im=imread(fullfile(im_directory,images(frame,1).name));
    now_pos = zeros(m,2);
    now_sz = zeros(m,2);
    for i=1:m
        now_pos(i,:) = floor([infor(i,1) infor(i,2)]);
        now_sz(i,:) = floor([infor(i,3) infor(i,4)]);
    end

    %% model transformation and matching
    if frame > 1
        for t1 = 1:pre_m
            for t2 = 1:m
                scores(t1,t2,frame) = patchmatch(pre_im,pre_pos(t1,:),pre_sz(t1,:),now_im,now_pos(t2,:),now_sz(t2,:),non_compressed_features, compressed_features, w2c);
            end
            [resulttmp(1,t1,frame),tm] = max(scores(t1,:,frame));
            if resulttmp(1,t1,frame) > params.eta1
                if ~results(1,tm,frame)
                    results(1,tm,frame) = t1;
                    labels(1,tm,frame) = labels(1,t1,frame-1);
                else
                    tn = results(1,tm,frame);
                    tmp = scores(tn,tm,frame);
                    if resulttmp(1,t1,frame) > tmp
                        results(1,tm,frame) = t1;
                        labels(1,tm,frame) = labels(1,t1,frame-1);
                    end
                end
            end
           %% match virtual objects
            virscore{frame} = zeros(vm,m);
            vtms = zeros(1,vm);
            vtmn = 1;
            if vm ~=0
                for t2 = 1:m
                    if ~results(1,t2,frame)
                        for ii = 1:vm
                            frameno = virobjects(ii,5);
                            vir_im = imread(fullfile(im_directory,images(frameno,1).name));
                            virscore{frame}(ii,t2) = pmatch(vir_im,virobjects(ii,:),now_im,now_pos(t2,:),now_sz(t2,:),non_compressed_features, compressed_features, w2c);
                        end
                    end
                    [vrt,vtm] = max(virscore{frame}(:,t2));
                    if vrt > params.zeta
                        if ~find(vtms==vtm)
                            results(1,t2,frame) = min(max(results(1,:,frame)+1,object_pnum));
                            labels(1,t2,frame) = virobjects(vtm,7);
                            vtms(vtmn) = vtm;
                            vtmn = vtmn +1;
                        else
                            v11= find(labels(1,:,frame)==virobjects(vtm,7));
                            if virscore{frame}(vtm,t2) > virscore{frame}(vtm,v11)
                                results(1,t2,frame) = min(max(results(1,:,frame)+1,object_pnum));
                                labels(1,t2,frame) = virobjects(vtm,7);
                                results(1,v11,frame) = 0;
                                labels(1,v11,frame) = 0;
                            end
                        end
                    end
                end
            end
        end
       %% build virtual objects
        t1n = pre_m;
        pre_m = m; 
        for t1 = 1:t1n
            if  max(scores(t1,:,frame)) < params.eta2
                if labels(1,t1,frame-1)
                    [det,obj,ff]= missdetection(now_im,pre_im,pre_pos(t1,:),pre_sz(t1,:),non_compressed_features, compressed_features, w2c);
                    if ff
                        objects{frame} = cat(1,objects{frame},obj);
                        detections{frame} = cat(1,detections{frame},det);
                        infor = cat(1,infor,obj);
                        now_pos =cat(1,now_pos,[obj(1),obj(2)]);
                        now_sz = cat(1,now_sz,[obj(3),obj(4)]);
                        if m >= object_pnum
                            fprintf('wrong');
                        else
                            scores(t1,m+1,frame) = det(5);
                            results(1,m+1,frame) = t1;
                            labels(1,m+1,frame) = labels(1,t1,frame-1);
                            m= m+1;
                        end
                    else
                        if labels(1,t1,frame-1)
                            if isempty(virobjects)
                                alternative = altersave(now_im,frame,pre_im,pre_pos(t1,:),pre_sz(t1,:),t1,labels(1,t1,frame-1),virflag,non_compressed_features, compressed_features, w2c);
                                virobjects = alternative;
                                virflag = virflag + 1;
                            else
                                alternative = altersave(now_im,frame,pre_im,pre_pos(t1,:),pre_sz(t1,:),t1,labels(1,t1,frame-1),virflag,non_compressed_features, compressed_features, w2c);
                                virobjects = cat(1,virobjects,alternative);
                                virflag = virflag + 1;
                            end
                        end
                    end
                end
            end
        end
       %% clear overlap region
        m1 = size(objects{frame},1);
        overlap =zeros(m1,m1);
        for tt1 = 1:m1
            for tt2 =1:m1
                overlap(tt1,tt2) =com_overlap(objects{frame}(tt1,1:2),objects{frame}(tt2,1:2),objects{frame}(tt1,3:4),objects{frame}(tt2,3:4));
            end
        end
        for iii =1:m
            if ~results(1,iii,frame)
                if max(scores(:,iii,frame)) < params.eta2 && detections{frame}(iii,5) > params.sigma
                    labels(1,iii,frame) = flag;
                    flag = flag + 1;
                end
            end
        end 
    end
end
flag = flag -1 ;
end