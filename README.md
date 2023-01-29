# 3D Scanning Final Project - Stereo Reconstruction

## References

* [Weekly Report](https://docs.google.com/document/d/1K6K0ElHKk27aSyPWNIXJ57GBT3060mLvXEGucMk_U0U/edit)
* TA's feedback

> Your proposal contains all necessary technical steps to solve the problem.
You want to merge the 3D output of multiple stereo reconstructions into a joint 3D model using ICP. We encourage you to do this, because it is an interesting next step, but note that it is beyond the standard pipeline for stereo reconstruction.
Please specify more concretely which “matching cost computation methods, cost (support) aggregation methods and disparity computation/optimization methods” you plan to compare. I encourage you to update the milestones accordingly, e.g. you condensed 3 weeks into one high-level bullet-point.
While you do not need to implement everything yourself (e.g. sparse keypoint detection/matching can be taken from OpenCV), please note that the whole project should not be implemented by simply calling OpenCV functions for everything. For instance, you could implement the dense matching yourself, since you also want to compare different implementations in your final report.
Typically, people start with calibrated cameras as input in their datasets to simplify the pre-processing. I think your datasets offer intrinsics already, so I suggest you use them, instead of calibrating the cameras yourself.

## Running the code

* See defined macros in `main.cpp` and `Libs/Pipeline.h` to decide the whole running procedure.

* See `CMakeLists.txt` for libraries installation.
