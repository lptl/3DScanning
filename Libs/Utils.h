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

void printCamParams(struct cameraParams camParams){
  std::cout << "left camera matrix:" << std::endl;
  std::cout << camParams.left_camera_matrix << " " << camParams.left_camera_matrix.size() << std::endl;
  std::cout << "right camera matrix:" << std::endl;
  std::cout << camParams.right_camera_matrix << " " << camParams.right_camera_matrix.size() << std::endl;
  std::cout << "left distortion coefficients:" << std::endl;
  std::cout << camParams.left_distortion_coeffs << " " << camParams.left_distortion_coeffs.size() << std::endl;
  std::cout << "right distortion coefficients:" << std::endl;
  std::cout << camParams.right_distortion_coeffs << " " << camParams.right_distortion_coeffs.size() << std::endl;
  std::cout << "left to right rotation:" << std::endl;
  std::cout << camParams.left_to_right_R << " " << camParams.left_to_right_R.size() << std::endl;
  std::cout << "left to right translation:" << std::endl;
  std::cout << camParams.left_to_right_T << " " << camParams.left_to_right_T.size() << std::endl;
  std::cout << "right to left rotation:" << std::endl;
  std::cout << camParams.right_to_left_R << " " << camParams.right_to_left_R.size() << std::endl;
  std::cout << "right to left translation:" << std::endl;
  std::cout << camParams.right_to_left_T << " " << camParams.right_to_left_T.size() << std::endl;
  return;
}

void readMatrixFromFile(std::string filename, int rows, int cols, cv::Mat& matrix){
  std::ifstream infile(filename);
  std::string line;
  int line_number = 0;
  while(std::getline(infile, line)){
    std::istringstream iss(line);
    for(int i = 0; i < cols; i++)
      iss >> matrix.at<double>(line_number, i);
    line_number++;
  }
  return;
}

void getCameraParamsBricks(std::string color_intrinsics, std::string depth_intrinsics, std::string left, std::string right, struct cameraParams *camParams){
  cv::Mat color_intrinsic_matrix(4, 4, CV_64F, cv::Scalar::all(0));
  cv::Mat depth_intrinsic_matrix(4, 4, CV_64F, cv::Scalar::all(0));
  cv::Mat left_extrinsic_matrix(4, 4, CV_64F, cv::Scalar::all(0));
  cv::Mat right_extrinsic_matrix(4, 4, CV_64F, cv::Scalar::all(0));

  readMatrixFromFile(color_intrinsics, 4, 4, color_intrinsic_matrix);
  readMatrixFromFile(depth_intrinsics, 4, 4, depth_intrinsic_matrix);
  readMatrixFromFile(left, 4, 4, left_extrinsic_matrix);
  readMatrixFromFile(right, 4, 4, right_extrinsic_matrix);

  cv::Mat color_intrinsic(3, 3, CV_64F, cv::Scalar::all(0));
  cv::Mat depth_intrinsic(3, 3, CV_64F, cv::Scalar::all(0));
  
  color_intrinsic.at<double>(0, 0) = color_intrinsic_matrix.at<double>(0, 0);
  color_intrinsic.at<double>(0, 1) = color_intrinsic_matrix.at<double>(0, 1);
  color_intrinsic.at<double>(0, 2) = color_intrinsic_matrix.at<double>(0, 2);
  color_intrinsic.at<double>(1, 0) = color_intrinsic_matrix.at<double>(1, 0);
  color_intrinsic.at<double>(1, 1) = color_intrinsic_matrix.at<double>(1, 1);
  color_intrinsic.at<double>(1, 2) = color_intrinsic_matrix.at<double>(1, 2);
  color_intrinsic.at<double>(2, 0) = color_intrinsic_matrix.at<double>(2, 0);
  color_intrinsic.at<double>(2, 1) = color_intrinsic_matrix.at<double>(2, 1);
  color_intrinsic.at<double>(2, 2) = color_intrinsic_matrix.at<double>(2, 2);
  
  depth_intrinsic.at<double>(0, 0) = depth_intrinsic_matrix.at<double>(0, 0);
  depth_intrinsic.at<double>(0, 1) = depth_intrinsic_matrix.at<double>(0, 1);
  depth_intrinsic.at<double>(0, 2) = depth_intrinsic_matrix.at<double>(0, 2);
  depth_intrinsic.at<double>(1, 0) = depth_intrinsic_matrix.at<double>(1, 0);
  depth_intrinsic.at<double>(1, 1) = depth_intrinsic_matrix.at<double>(1, 1);
  depth_intrinsic.at<double>(1, 2) = depth_intrinsic_matrix.at<double>(1, 2);
  depth_intrinsic.at<double>(2, 0) = depth_intrinsic_matrix.at<double>(2, 0);
  depth_intrinsic.at<double>(2, 1) = depth_intrinsic_matrix.at<double>(2, 1);
  depth_intrinsic.at<double>(2, 2) = depth_intrinsic_matrix.at<double>(2, 2);
  

  camParams->left_camera_matrix = color_intrinsic;
  camParams->right_camera_matrix = color_intrinsic;
  camParams->fX = 1170.19;
  camParams->fY = 1170.19;
  camParams->cX = 647.75;
  camParams->cY = 483.75;

  //TODO: what about the baseline
  camParams->baseline = 0.534;
  camParams->left_distortion_coeffs = cv::Mat::zeros(5, 1, CV_32F);
  camParams->right_distortion_coeffs = cv::Mat::zeros(5, 1, CV_32F);

  cv::Mat left_to_right = right_extrinsic_matrix.inv() * left_extrinsic_matrix;
  camParams->left_to_right_R = left_to_right(cv::Rect(0, 0, 3, 3));
  camParams->left_to_right_T = left_to_right(cv::Rect(3, 0, 1, 3));
  cv::Mat right_to_left = left_extrinsic_matrix.inv() * right_extrinsic_matrix;
  camParams->right_to_left_R = right_to_left(cv::Rect(0, 0, 3, 3));
  camParams->right_to_left_T = right_to_left(cv::Rect(3, 0, 1, 3));

  camParams->empty = false;
  //  printCamParams(*camParams);
  return;
}

void getCameraParamsKITTI(std::string calib_file, struct cameraParams *camParams)
{
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
        else
        {
            continue;
        }
    }

    // calculate the left-to-right rotation and translation and right-to-left rotation and translation
    cv::Mat left_extrinsics = cv::Mat::eye(4, 4, CV_64FC1);
    cv::Mat right_extrinsics = cv::Mat::eye(4, 4, CV_64FC1);
    left_R.copyTo(left_extrinsics(cv::Rect(0, 0, 3, 3)));
    left_T.copyTo(left_extrinsics(cv::Rect(3, 0, 1, 3)));
    right_R.copyTo(right_extrinsics(cv::Rect(0, 0, 3, 3)));
    right_T.copyTo(right_extrinsics(cv::Rect(3, 0, 1, 3)));
    cv::Mat left_to_right = right_extrinsics * left_extrinsics.inv();
    camParams->left_to_right_R = left_to_right(cv::Rect(0, 0, 3, 3));
    camParams->left_to_right_T = left_to_right(cv::Rect(3, 0, 1, 3));
    cv::Mat right_to_left = left_extrinsics * right_extrinsics.inv();
    camParams->right_to_left_R = right_to_left(cv::Rect(0, 0, 3, 3));
    camParams->right_to_left_T = right_to_left(cv::Rect(3, 0, 1, 3));

    // TODO: Calculate these values instead of hardcoding
    camParams->baseline = 0.5327190420453419;
    camParams->fX = 721.5377;
    camParams->fY = 721.5377;
    camParams->cX = 609.5593;
    camParams->cY = 172.854;

    camParams->empty = false;
    // printCamParams(*camParams);

    return;
}

bool writeMesh(Vertex *vertices, unsigned int ImageWidth, unsigned int ImageHeight, const std::string &filename, float edgeThreshold = 0.01f)
{
    unsigned int nVertices = ImageWidth * ImageHeight;
    unsigned int nTriangles = 0;
    std::vector<Vector3i> FaceId;

    for (unsigned int i = 0; i < ImageHeight - 1; i++)
    {
        for (unsigned int j = 0; j < ImageWidth - 1; j++)
        {
            unsigned int i0 = i * ImageWidth + j;
            unsigned int i1 = (i + 1) * ImageWidth + j;
            unsigned int i2 = i * ImageWidth + j + 1;
            unsigned int i3 = (i + 1) * ImageWidth + j + 1;

            bool valid0 = vertices[i0].position.allFinite();
            bool valid1 = vertices[i1].position.allFinite();
            bool valid2 = vertices[i2].position.allFinite();
            bool valid3 = vertices[i3].position.allFinite();

            if (valid0 && valid1 && valid2)
            {
                float d0 = (vertices[i0].position - vertices[i1].position).norm();
                float d1 = (vertices[i0].position - vertices[i2].position).norm();
                float d2 = (vertices[i1].position - vertices[i2].position).norm();
                if (d0 < edgeThreshold && d1 < edgeThreshold && d2 < edgeThreshold)
                {
                    Vector3i faceIndices(i0, i1, i2);
                    FaceId.push_back(faceIndices);
                    nTriangles++;
                }
            }

            if (valid1 && valid2 && valid3)
            {
                float d0 = (vertices[i3].position - vertices[i1].position).norm();
                float d1 = (vertices[i3].position - vertices[i2].position).norm();
                float d2 = (vertices[i1].position - vertices[i2].position).norm();
                if (d0 < edgeThreshold && d1 < edgeThreshold && d2 < edgeThreshold)
                {
                    Vector3i faceIndices(i1, i2, i3);
                    FaceId.push_back(faceIndices);
                    nTriangles++;
                }
            }
        }
    }

    // Write off file
    std::ofstream outFile(filename);
    if (!outFile.is_open())
        return false;

    // Write header.
    outFile << "COFF" << std::endl;
    outFile << nVertices << " " << nTriangles << " 0" << std::endl;

    // Save vertices.
    for (unsigned int i = 0; i < nVertices; i++)
    {
        const auto &vertex = vertices[i];
        if (vertex.position.allFinite())
            outFile << vertex.position.x() << " " << vertex.position.y() << " " << vertex.position.z() << " "
                    << int(vertex.color.x()) << " " << int(vertex.color.y()) << " " << int(vertex.color.z()) << " " << int(vertex.color.w()) << std::endl;
        else
            outFile << "0.0 0.0 0.0 0 0 0 0" << std::endl;
    }
    // Save faces.
    for (Vector3i &faceIndices : FaceId)
    {
        outFile << "3 " << faceIndices[0] << " " << faceIndices[1] << " " << faceIndices[2] << std::endl;
    }

    // Close file.
    outFile.close();
    return true;
}
