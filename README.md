# 3D Scanning Final Project - Stereo Reconstruction

## References

* [Weekly Report](https://docs.google.com/document/d/1K6K0ElHKk27aSyPWNIXJ57GBT3060mLvXEGucMk_U0U/edit)
* TA's feedback

## Project's overview
This project proposes a pipeline for comparing different stereo reconstruction methods and has already been tested on two datasets with successful results.  The pipeline involves the following steps: camera calibration to remove image distortion, key points extraction and matching to rectify images, dense stereo matching to calculate disparity maps and obtain depth information, and final 3D model generation from the calculated depth maps.

The project will implement various stereo-matching algorithms such as StereoSGBM and StereoBM and perform post-filtering to increase the quality of the disparity map. The 3D points will be calculated from the stereo correspondences and disparity values, and then merged into a complete 3D model using ICP or an improved algorithm. The aim of this project is to compare the performance of different stereo reconstruction methods and obtain the optimal solution for 3D modelling.
![Pipeline Review](pipeline_neu.png)

## Running the code

* See defined macros in `main.cpp` and `Libs/Pipeline.h` to decide the whole running procedure.



* See `CMakeLists.txt` for libraries installation.
