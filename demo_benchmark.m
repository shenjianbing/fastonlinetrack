clc;clear;close all;

seqs = {
        'ADL-Rundle-1','ADL-Rundle-3','AVG-TownCentre',...
        'ETH-Crossing','ETH-Jelmoli','ETH-Linthescher',...
        'KITTI-16','KITTI-19','PETS09-S2L2','TUD-Crossing',...
        'Venice-1'   
        }; %11 sequences
    
for s = 1:length(seqs)
    clearvars -except seqs s
    single_tracker(seqs{s});
end
