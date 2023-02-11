#pragma once

#include <string>
#include <regex>
#include <vector>
#include <fstream>
#include <sstream>

#include "Eigen.h"
#include "Types.h"

struct filenameType extract_file_name(std::string filename)
{
    filenameType type;
    type.name = filename;
    type.number = std::stoi(std::regex_replace(filename, std::regex("[a-zA-Z_-]"), ""));
    if (filename.find("color") != std::string::npos)
        type.category = 0;
    else if (filename.find("depth") != std::string::npos)
        type.category = 1;
    else if (filename.find("pose") != std::string::npos)
        type.category = 2;
    else
        type.category = -1;
    return type;
}

std::string get_file_name(int number, int category)
{
    if (number >= 772)
        number = 770;
    std::string number_string = std::to_string(number);
    if (number < 10)
        number_string = "00" + number_string;
    else if (number < 100)
        number_string = "0" + number_string;
    std::string filename = "frame-000" + number_string + ".";
    if (category == 0)
        filename += "color.png";
    else if (category == 1)
        filename += "depth.png";
    else if (category == 2)
        filename += "pose.txt";
    return filename;
}

bool compare_string(std::string str1, std::string str2)
{
    if (str1.compare(str2) == 0)
        return true;
    else
        return false;
}

std::string type2str(int type) {
    std::string r;

    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch (depth) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
    }

    r += "C";
    r += (chans + '0');

    return r;
}

void getCameraParamsKITTI(std::string calib_file, struct cameraParams *camParams)
{
    std::ifstream infile(calib_file);
    std::string line;

    cv::Mat left_R(3, 3, CV_64FC1, cv::Scalar::all(0));
    cv::Mat right_R(3, 3, CV_64FC1, cv::Scalar::all(0));

    cv::Mat left_T(3, 1, CV_64FC1, cv::Scalar::all(0));
    cv::Mat right_T(3, 1, CV_64FC1, cv::Scalar::all(0));

    cv::Mat left_P_rect(3, 4, CV_64FC1, cv::Scalar::all(0));
    cv::Mat right_P_rect(3, 4, CV_64FC1, cv::Scalar::all(0));

    while (std::getline(infile, line))
    {
        std::istringstream iss(line);
        std::string param, data;
        iss >> param;

        if (compare_string(param, "K_02:"))
        {
            cv::Mat left_cam(3, 3, CV_64FC1, cv::Scalar::all(0));
            for (size_t i = 0; i < 3; i++)
            {
                for (size_t j = 0; j < 3; j++)
                {
                    iss >> left_cam.at<double>(i, j);
                }
            }
            camParams->left_camera_matrix = left_cam;
        }
        else if (compare_string(param, "D_02:"))
        {
            cv::Mat left_distort(5, 1, CV_64FC1, cv::Scalar::all(0));
            for (size_t i = 0; i < 5; i++)
            {
                iss >> left_distort.at<double>(i, 0);
            }
            camParams->left_distortion_coeffs = left_distort;
        }
        else if (compare_string(param, "R_02:"))
        {
            for (size_t i = 0; i < 3; i++)
            {
                for (size_t j = 0; j < 3; j++)
                {
                    iss >> left_R.at<double>(i, j);
                }
            }
        }
        else if (compare_string(param, "T_02:"))
        {
            for (size_t i = 0; i < 3; i++)
            {
                iss >> left_T.at<double>(i, 0);
            }
        }
        else if (compare_string(param, "P_rect_02:"))
        {
            for (size_t i = 0; i < 3; i++)
            {
                for (size_t j = 0; j < 4; j++)
                {
                    iss >> left_P_rect.at<double>(i, j);
                }
            }
        }
        else if (compare_string(param, "K_03:"))
        {
            cv::Mat right_cam(3, 3, CV_64FC1, cv::Scalar::all(0));
            for (size_t i = 0; i < 3; i++)
            {
                for (size_t j = 0; j < 3; j++)
                {
                    iss >> right_cam.at<double>(i, j);
                }
            }
            camParams->right_camera_matrix = right_cam;
        }
        else if (compare_string(param, "D_03:"))
        {
            cv::Mat right_distort(5, 1, CV_64FC1, cv::Scalar::all(0));
            for (size_t i = 0; i < 5; i++)
            {
                iss >> right_distort.at<double>(i, 0);
            }
            camParams->right_distortion_coeffs = right_distort;
        }
        else if (compare_string(param, "R_03:"))
        {
            for (size_t i = 0; i < 3; i++)
            {
                for (size_t j = 0; j < 3; j++)
                {
                    iss >> right_R.at<double>(i, j);
                }
            }
        }
        else if (compare_string(param, "T_03:"))
        {
            for (size_t i = 0; i < 3; i++)
            {
                iss >> right_T.at<double>(i, 0);
            }
        }
        else if (compare_string(param, "P_rect_03:"))
        {
            for (size_t i = 0; i < 3; i++)
            {
                for (size_t j = 0; j < 4; j++)
                {
                    iss >> right_P_rect.at<double>(i, j);
                }
            }
        }
        else
        {
            continue;
        }
    }

    camParams->left_to_right_R = left_R.t() * right_R;
    camParams->left_to_right_T = (left_R.t() * right_T) - left_T;

    camParams->fX = left_P_rect.at<double>(0, 0);
    camParams->cX = left_P_rect.at<double>(0, 2);
    camParams->fY = left_P_rect.at<double>(1, 1);
    camParams->cY = left_P_rect.at<double>(1, 2);
    camParams->baseline = ((left_P_rect.at<double>(0, 3) - right_P_rect.at<double>(0, 3)) - (left_P_rect.at<double>(3, 3) - right_P_rect.at<double>(3, 3))) / camParams->fX;

    camParams->empty = false;

    return;
}


bool writeMesh(Vertex* vertices, unsigned int width, unsigned int height, const std::string& filename, float edgeThreshold = 0.01f)
{
    unsigned int nVertices = width * height;
    unsigned nFaces = 0;
    std::vector<Vector3i> Face_idxs;

    for (int v = 0; v < height - 1; v++) {
        for (int u = 0; u < width - 1; u++) {
            int idx_0 = v * width + u;
            int idx_1 = (v + 1) * width + u;
            int idx_2 = v * width + u + 1;
            int idx_3 = (v + 1) * width + u + 1;

            bool valid_0 = (vertices[idx_0].position[0] != MINF);
            bool valid_1 = (vertices[idx_1].position[0] != MINF);
            bool valid_2 = (vertices[idx_2].position[0] != MINF);
            bool valid_3 = (vertices[idx_3].position[0] != MINF);

            if (valid_0 && valid_1 && valid_2) {
                Vector4f p_0 = vertices[idx_0].position;
                Vector4f p_1 = vertices[idx_1].position;
                Vector4f p_2 = vertices[idx_2].position;
                float d_01 = (p_0 - p_1).norm();
                float d_02 = (p_0 - p_2).norm();
                float d_12 = (p_1 - p_2).norm();
                if (d_01 < edgeThreshold && d_02 < edgeThreshold && d_12 < edgeThreshold)
                {
                    Vector3i Face_idx(idx_0, idx_1, idx_2);
                    Face_idxs.push_back(Face_idx);
                    nFaces++;
                }
            }
            if (valid_3 && valid_1 && valid_2) {
                Vector4f p_3 = vertices[idx_3].position;
                Vector4f p_1 = vertices[idx_1].position;
                Vector4f p_2 = vertices[idx_2].position;
                float d_31 = (p_3 - p_1).norm();
                float d_32 = (p_3 - p_2).norm();
                float d_12 = (p_1 - p_2).norm();
                if (d_31 < edgeThreshold && d_32 < edgeThreshold && d_12 < edgeThreshold)
                {
                    Vector3i Face_idx(idx_1, idx_2, idx_3);
                    Face_idxs.push_back(Face_idx);
                    nFaces++;
                }
            }
        }
    }


    // Write off file
    std::ofstream outFile(filename);
    if (!outFile.is_open()) return false;

    // write header
    outFile << "COFF" << std::endl;
    outFile << nVertices << " " << nFaces << " 0" << std::endl;

    // TODO: save vertices
    for (int idx = 0; idx < nVertices; idx++) {
        if ((vertices + idx)->position[0] == MINF) outFile << "0.0 0.0 0.0 ";
        else
            outFile << (vertices + idx)->position[0] << " "\
            << (vertices + idx)->position[1] << " "\
            << (vertices + idx)->position[2] << " ";

        outFile << int((vertices + idx)->color[0]) << " "\
            << int((vertices + idx)->color[1]) << " "\
            << int((vertices + idx)->color[2]) << " "\
            << int((vertices + idx)->color[3]) << std::endl;
    }

    for (Vector3i& Face_idx : Face_idxs)
        outFile << "3 " << Face_idx[0] << " " << Face_idx[1] << " " << Face_idx[2] << " " << std::endl;

    // close file
    outFile.close();

    return true;
}

cv::Vec3d rot2euler(cv::Mat& R)
{
    double sy = sqrt(R.at<double>(0, 0) * R.at<double>(0, 0) + R.at<double>(1, 0) * R.at<double>(1, 0));

    bool singular = sy < 1e-6;

    double x, y, z;
    if (!singular)
    {
        x = atan2(R.at<double>(2, 1), R.at<double>(2, 2));
        y = atan2(-R.at<double>(2, 0), sy);
        z = atan2(R.at<double>(1, 0), R.at<double>(0, 0));
    }
    else
    {
        x = atan2(-R.at<double>(1, 2), R.at<double>(1, 1));
        y = atan2(-R.at<double>(2, 0), sy);
        z = 0;
    }

    return cv::Vec3d(x, y, z);
}

double euler_mse(cv::Mat R1, cv::Mat R2)
{

    // Convert rotation matrices to Euler angles
    cv::Vec3f euler1, euler2;
    euler1 = rot2euler(R1);
    euler2 = rot2euler(R2);

    // Calculate mean squared error
    double mse = 0;
    mse = cv::norm(euler1, euler2, cv::NORM_L2);

    mse /= 3;
    return mse;
}

double disp_error(cv::Mat gt_disp, cv::Mat calc_disp, cv::Mat &err_img)
{
    cv::Mat err;
    cv::absdiff(gt_disp, calc_disp, err);

    int total = 0, err_count = 0;
    for (int i = 0; i < err.rows; i++)
    {
        for (int j = 0; j < err.cols; j++)
        {
            float val = err.at<float>(i, j);
            if (val > 0)
            {
                total++;
                if (val > 3 && val / gt_disp.at<float>(i, j) > 0.05) { err_count++; }
            }
        }
    }

    err.convertTo(err_img, CV_8UC1);
    cv::applyColorMap(err_img, err_img, cv::COLORMAP_INFERNO);
    return err_count / total;
}