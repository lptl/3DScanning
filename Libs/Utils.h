#pragma once

#include <string>
#include <regex>
#include <vector>
#include <fstream>
#include <sstream>

#include "Eigen.h"
#include "Types.h"

struct filenameType extract_file_name(std::string filename){
    filenameType type;
    type.name = filename;
    type.number = std::stoi(std::regex_replace(filename, std::regex("[a-zA-Z_-]"), ""));
    if(filename.find("color") != std::string::npos)
        type.category = 0;
    else if(filename.find("depth") != std::string::npos)
        type.category = 1;
    else if(filename.find("pose") != std::string::npos)
        type.category = 2;
    else
        type.category = -1;
    return type;
}


std::string get_file_name(int number, int category){
    if(number >= 772)
        number = 770;
    std::string number_string = std::to_string(number);
    if(number < 10)
        number_string = "00" + number_string;
    else if(number < 100)
        number_string = "0" + number_string;
    std::string filename = "frame-000" + number_string + ".";
    if(category == 0)
        filename += "color.png";
    else if(category == 1)
        filename += "depth.png";
    else if(category == 2)
        filename += "pose.txt";
    return filename;
}

bool compare_string(std::string str1, std::string str2){
    if(str1.compare(str2) == 0)
        return true;
    else
        return false;
}

void getIntrinsics(std::string calib_file, struct intrinsics *intrs) {
    std::ifstream infile(calib_file);
    std::string line;

    cv::Mat left_R(3, 3, CV_64FC1, cv::Scalar::all(0));
    cv::Mat left_T(3, 1, CV_64FC1, cv::Scalar::all(0));
    cv::Mat right_R(3, 3, CV_64FC1, cv::Scalar::all(0));
    cv::Mat right_T(3, 1, CV_64FC1, cv::Scalar::all(0));

    while (std::getline(infile, line))
    {
        std::istringstream iss(line);
        std::string param, data;
        iss >> param;

        if (compare_string(param, "K_02:")) {
            cv::Mat left_cam(3, 3, CV_64FC1, cv::Scalar::all(0));
            for (size_t i = 0; i < 3; i++) {
                for (size_t j = 0; j < 3; j++) {
                    iss >> left_cam.at<double>(i, j);
                }
            }
            intrs->left_camera_matrix = left_cam;
        }
        else if (compare_string(param, "D_02:")) {
            cv::Mat left_distort(5, 1, CV_64FC1, cv::Scalar::all(0));
            for (size_t i = 0; i < 5; i++) {
                iss >> left_distort.at<double>(i, 0);
            }
            intrs->left_distortion_coeffs = left_distort;
        }
        else if (compare_string(param, "R_02:")) {
            for (size_t i = 0; i < 3; i++) {
                for (size_t j = 0; j < 3; j++) {
                    iss >> left_R.at<double>(i, j);
                }
            }
        }
        else if (compare_string(param, "T_02:")) {
            for (size_t i = 0; i < 3; i++) {
                iss >> left_T.at<double>(i, 0);
            }
        }
        else if (compare_string(param, "K_03:")) {
            cv::Mat right_cam(3, 3, CV_64FC1, cv::Scalar::all(0));
            for (size_t i = 0; i < 3; i++) {
                for (size_t j = 0; j < 3; j++) {
                    iss >> right_cam.at<double>(i, j);
                }
            }
            intrs->right_camera_matrix = right_cam;
        }
        else if (compare_string(param, "D_03:")) {
            cv::Mat right_distort(5, 1, CV_64FC1, cv::Scalar::all(0));
            for (size_t i = 0; i < 5; i++) {
                iss >> right_distort.at<double>(i, 0);
            }
            intrs->right_distortion_coeffs = right_distort;
        }
        else if (compare_string(param, "R_03:")) {
            for (size_t i = 0; i < 3; i++) {
                for (size_t j = 0; j < 3; j++) {
                    iss >> right_R.at<double>(i, j);
                }
            }
        }
        else if (compare_string(param, "T_03:")) {
            for (size_t i = 0; i < 3; i++) {
                iss >> right_T.at<double>(i, 0);
            }
        }
        else {
            continue;
        }
    }

    intrs->left_to_right_R = left_R.inv() * right_R;
    intrs->left_to_right_T = right_T - left_T;
    intrs->empty = true;

    return;
}