# 3D Scanning Final Project - Stereo Reconstruction

## References

* [Weekly Report](https://docs.google.com/document/d/1K6K0ElHKk27aSyPWNIXJ57GBT3060mLvXEGucMk_U0U/edit)
* TA's feedback

> Your proposal contains all necessary technical steps to solve the problem.
You want to merge the 3D output of multiple stereo reconstructions into a joint 3D model using ICP. We encourage you to do this, because it is an interesting next step, but note that it is beyond the standard pipeline for stereo reconstruction.
Please specify more concretely which “matching cost computation methods, cost (support) aggregation methods and disparity computation/optimization methods” you plan to compare. I encourage you to update the milestones accordingly, e.g. you condensed 3 weeks into one high-level bullet-point.
While you do not need to implement everything yourself (e.g. sparse keypoint detection/matching can be taken from OpenCV), please note that the whole project should not be implemented by simply calling OpenCV functions for everything. For instance, you could implement the dense matching yourself, since you also want to compare different implementations in your final report.
Typically, people start with calibrated cameras as input in their datasets to simplify the pre-processing. I think your datasets offer intrinsics already, so I suggest you use them, instead of calibrating the cameras yourself.

## Questions

* opencv only provides feature matcher, which use feature descriptors as input. There is no keypoint matcher in opencv. But harris keypoint matcher produce keypoints and harris corner responses.

* Why the matching result of feature matcher contains the index of keypoints

* still confusing about how to merge multiple point clouds to get a whole 3D model because ICP seems not to do the job. One solution is each time merge two models and then get the final model.
