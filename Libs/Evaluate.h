#pragma once
#include <iostream>
#include <dirent.h>
#include <algorithm>
#include "Pipeline.h"
#include <cassert>

template<typename Func>
double computeRunTime(Func f){

    int64 start = cv::getTickCount();

    f();

    int64 end = cv::getTickCount();
    double runtime = (end - start) / cv::getTickFrequency();
    std::cout << "Runtime: " << runtime << " seconds" << std::endl;
    std::cout << "Stereo Reconstruction Finished" << std::endl;
    return runtime;
}

std::vector<struct detectResult> processXimages(int image_number,std::string dataset_dir){
    DIR* directory = opendir(dataset_dir.c_str());
    struct dirent* entry;
    std::vector<struct detectResult> reusltVector;
    int readCount = 0;
    if(directory == NULL){
        std::cout << "Can not read file from the dataset directory" << std::endl;
        return reusltVector;
    } else {
        while((entry = readdir(directory)) != NULL && readCount < image_number){
            if(entry->d_name[0] != 'f')
                continue;
            struct filenameType filename_type = extract_file_name(entry->d_name);
            if(filename_type.category != 0)
                continue;
            std::string filename = get_file_name(filename_type.number, filename_type.category);
            std::cout << "Processing image " << filename << std::endl;
            cv::Mat img = cv::imread(dataset_dir + filename, cv::IMREAD_COLOR);
            struct detectResult result;
            detect_keypoints_or_features(filename, img,&result);
            reusltVector.push_back(result);
            readCount++;
        }
        closedir(directory);
    }
    return reusltVector;
}

bool is_rotation_matrix(cv::Mat& R)
{
    cv::Mat Rt;
    transpose(R, Rt);
    cv::Mat shouldBeIdentity = Rt * R;
    cv::Mat I = cv::Mat::eye(3, 3, shouldBeIdentity.type());

    return  cv::norm(I, shouldBeIdentity) < 1e-6;

}

cv::Vec3d rotation_matrix2euler_angles(cv::Mat& R)
{

    if (!is_rotation_matrix(R)) {
        std::cout << "The input Matrix is not a rotation matrix!" << std::endl;
        return cv::Vec3d();
    }

    double sy = sqrt(R.at<double>(0, 0) * R.at<double>(0, 0) + R.at<double>(1, 0) * R.at<double>(1, 0));

    bool singular = sy < 1e-6; // If

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

    std::cout << "Convert to euler angle: " << cv::Vec3d(x, y, z) << std::endl;
    return cv::Vec3d(x, y, z);

}

double euler_square_error(cv::Mat R1, cv::Mat R2) {

    // Convert rotation matrices to Euler angles
    cv::Vec3f euler1, euler2;
    euler1 = rotation_matrix2euler_angles(R1);
    euler2 = rotation_matrix2euler_angles(R2);

    // Calculate mean squared error
    double mse = 0;
    mse = cv::norm(euler1, euler2, cv::NORM_L2);
    //for (int i = 0; i < 3; i++) {
    //    mse += pow(euler1[i] - euler2[i], 2);
    //}
    mse /= 3;
    std::cout << "Mean Squared Error: " << mse << std::endl;
    return mse;
}

struct MSEextrinsic matching_method_compare(std::vector<cv::KeyPoint> keypoints1, std::vector<cv::KeyPoint> keypoints2, std::vector<cv::DMatch> good_matches, struct cameraParams camParams) {
    
    std::cout << "Evaluate "<< DESCRIPTOR_METHOD <<" description method........." << std::endl;
    std::vector<cv::Point2f> points1, points2;
    for (int i = 0; i < good_matches.size(); i++) {
        points1.push_back(keypoints1[good_matches[i].queryIdx].pt);
        points2.push_back(keypoints2[good_matches[i].trainIdx].pt);
    }

    //if (points1.size() != points2.size() || points1.size() < 5) {
    //    std::cout << "Error: The number of points for computing essential matrix is not enough." << std::endl;
    //    return 1;
    //}

    cv::Mat fundamental_matrix;
    find_fundamental_matrix(keypoints1, keypoints2, good_matches, fundamental_matrix);

    // Using essential matrix to compute rotation and translation matrix
    // cv::Mat essential_matrix = camParams.left_camera_matrix.t() * fundamental_matrix * camParams.right_camera_matrix;
    cv::Mat essential_matrix = cv::findEssentialMat(points1, points2, camParams.left_camera_matrix, camParams.left_distortion_coeffs, camParams.right_camera_matrix,camParams.right_distortion_coeffs);
    
    
    // Calculate Rotation matrix and tranlation matrix between two images
    cv::Mat R1, R2, T;
    cv::decomposeEssentialMat(essential_matrix, R1, R2, T);


    //cv::Mat ground_rvec,rvec1, rvec2;
    //cv::Rodrigues(R1, rvec1);
    //cv::Rodrigues(R2, rvec2);
    //cv::Rodrigues(camParams.left_to_right_R, ground_rvec);

    //double dis_R1 = cv::norm(ground_rvec, rvec1);
    //double dis_R2 = cv::norm(ground_rvec, rvec2);

    //double mse_R;
    //if (dis_R1 > dis_R2) {
    //    mse_R = dis_R2;
    //}
    //else {
    //    mse_R = dis_R1;
    //}
    struct MSEextrinsic mse_extr;
    double MSE_R, MSE_R1, MSE_R2;

    MSE_R1 = euler_square_error(camParams.left_to_right_R, R1);
    MSE_R2 = euler_square_error(camParams.left_to_right_R, R2);

    //Choose the smaller one as the final result
    if (MSE_R1 > MSE_R2) {
        MSE_R = MSE_R2;
    }
    else {
        MSE_R = MSE_R1;
    }

    //Calculate translation MSE value
    double MSE_T = cv::norm(camParams.left_to_right_T, T, cv::NORM_L2);

    std::cout << "mse_R: " << MSE_R << std::endl;
    std::cout << "mse_T: " << MSE_T << std::endl;

    std::cout << "Essential matrix" << essential_matrix << std::endl;
    std::cout << "R1 matrix" << R1 << std::endl;
    std::cout << "R2 matrix" << R2 << std::endl;
    std::cout << "T matrix" << T << std::endl;

    mse_extr.mse_R = MSE_R;
    mse_extr.mse_T = MSE_T;

    return mse_extr;
}