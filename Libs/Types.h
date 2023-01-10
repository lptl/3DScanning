#pragma once

#include "OpenCVLib.h"
#include "Eigen.h"

struct filenameType {
    int number = 0; // 000000, the number of the picture
    int category = -1; // 0: color, 1: depth, 2: pose
    std::string name = ""; // fullname
};

struct detectResult{
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    struct filenameType filetype;
};

struct Vertex
{
    //Eigen::EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // position stored as 4 floats (4th component is supposed to be 1.0)
    Eigen::Vector4f position;
    // color stored as 4 unsigned char
    //Vector4uc color;
};