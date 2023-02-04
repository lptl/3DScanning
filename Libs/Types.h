#pragma once

#include "OpenCVLib.h"

struct filenameType
{
    int number = 0;        // 000000, the number of the picture
    int category = -1;     // 0: color, 1: depth, 2: pose
    std::string name = ""; // fullname
};

struct detectResult
{
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    struct filenameType filetype;
};

struct cameraParams
{
    cv::Mat left_camera_matrix;
    cv::Mat left_distortion_coeffs;
    cv::Mat right_camera_matrix;
    cv::Mat right_distortion_coeffs;
    cv::Mat left_to_right_R;
    cv::Mat left_to_right_T;

    double baseline;
    double fX;
    double fY;
    double cX;
    double cY;

    bool empty = true;
};

struct MSEextrinsic
{
    double mse_R;
    double mse_T;
};