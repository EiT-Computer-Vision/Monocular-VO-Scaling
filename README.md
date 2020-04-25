# Monocular-VO-Scaling
Using already implemented visual odometry for monocular vision, determining the translation scale from known height of the camera above the horizontal ground plane.

Base code is taken from https://github.com/uoip/monoVO-python


SETUP TO RUN ON OWN COMPUTER 

1) Request download link from the link http://www.cvlibs.net/download.php?file=data_odometry_gray.zip
2) Download the grayscale visual odometry dataset
3) Download ground truth poses from http://www.cvlibs.net/download.php?file=data_odometry_poses.zip
for comparison and evaluation of performance
4) Set path to your directory path. Assumed structure is ../dataset/poses
and ../dataset/sequences from your project folder. 
5) Download the relevant libraries if not done (openCV).
6) Choose sequence #4 or #6 when testing the program in the test.py file.
7) Run test.py 
