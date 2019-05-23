This MATLAB code implements the algorithm of Fast Online Tracking with Detection Refinement. 
**************************************************************
[1] J. Shen, D. Yu, L. Deng, X. Dong, Fast online tracking with detection refinement, 
IEEE Trans. on Intelligent Transportation Systems, 19(1):162-173, 2018
**********Please cite our TITS18 paper, Thanks. **************
**************************************************************

Installation (Windows7 + Matlab 2014a)
To be able to run the "mexResize" function, try to use either one of the included mex-files 
or compile one of your own. OpenCV is needed for this. 

******************************************************
Run the run_single_tracker.m script to test the tracker.

>> demo_single_tracker
where 'Skater' could be replaced by any other video title. 

>> demo_benchmark for the whole benchmark

**********Please cite our related papers, Thanks. ****************************
[1] J. Shen, D. Yu, L. Deng, X. Dong, Fast online tracking with detection refinement, 
IEEE Trans. on Intelligent Transportation Systems, 19(1):162-173, 2018
[2] X. Dong, J. Shen, D. Yu, W. Wang, J. Liu, and H. Huang, Occlusion-aware real-time object tracking, 
IEEE Trans. on Multimedia, 19(4):763-771, 2017
********** Thanks. ********************* ****************************

Note that we also use the open source code for computing the HOG features.
