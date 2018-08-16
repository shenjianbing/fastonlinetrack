clc;clear;close all;
%% add all paths
addpath('./mex');
addpath('./matching');
addpath('./toolbox')

% tracking for single video;
clip_title = 'ETH-Crossing';
single_tracker(clip_title);
