#pragma once

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