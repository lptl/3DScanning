#include <iostream>
#include <fstream>
#include <array>
#include <dirent.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/hal/interface.h>
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/calib3d.hpp>

#include "Libs/Eigen.h"
#include "Libs/VirtualSensor.h"
#include "Libs/Types.h"

#define DESCRIPTOR_METHOD "sift" // harris, sift, surf, orb, brisk, kaze, akaze
#define DESCRIPTOR_MATCHING_METHOD "flann" // flann, brute_force
#define FUNDAMENTAL_MATRIX_METHOD "ransac" // ransac, lmeds, 7point, 8point

using namespace cv;
using namespace cv::xfeatures2d;

void detect_keypoints_or_features(Mat);  
struct filenameType extract_file_name(std::string);
std::string get_file_name(int, int);
void process_pair_images(std::string, std::string, std::string);
struct detectResult detect_keypoints_or_features(std::string, std::string, Mat);
std::vector<DMatch> match_descriptors(Mat, Mat, struct detectResult, struct detectResult);
Mat find_fundamental_matrix(std::vector<KeyPoint>, std::vector<KeyPoint>, std::vector<DMatch>);
void rectify_images(Mat, Mat, std::vector<KeyPoint>, std::vector<KeyPoint>, std::vector<DMatch>, Mat);

// TESTED
struct filenameType extract_file_name(std::string filename){
    filenameType type;
    type.name = filename;
    type.number = std::stoi(filename.substr(9, 12));
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

// TESTED
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

void process_pair_images(std::string dataset_dir, std::string filename1, std::string filename2){
    Mat img1 = imread(dataset_dir + filename1, IMREAD_COLOR);
    Mat img2 = imread(dataset_dir + filename2, IMREAD_COLOR);
    // TODO: calibrate image distortion if needed
    if(img1.empty() || img2.empty()){
        std::cout << "Error: Image not found or failed to open image." << std::endl;
        return;
    }
    // std::cout << "detecting keypoints and features for image " << filename1 << " and " << filename2 << std::endl;
    struct detectResult result1 = detect_keypoints_or_features(dataset_dir, filename1, img1);
    struct detectResult result2 = detect_keypoints_or_features(dataset_dir, filename2, img2);
    std::cout << "Processing " << filename1 << " and " << filename2 << std::endl;
    std::cout << "Matching descriptors for image " << std::endl;
    std::vector<DMatch> correspondences = match_descriptors(img1, img2, result1, result2);
    std::cout << "Finding fundamental matrix for image " << std::endl;
    Mat fundamental_matrix = find_fundamental_matrix(result1.keypoints, result2.keypoints, correspondences);
    // std::cout << fundamental_matrix << std::endl;
    std::cout << "Rectifying images for image " << std::endl;
    rectify_images(img1, img2, result1.keypoints, result2.keypoints, correspondences, fundamental_matrix);
    // use rectified images to do stereo matching
    // https://docs.opencv.org/3.4/d2/d6e/classcv_1_1StereoMatcher.html#a03f7087df1b2c618462eb98898841345
    std::cout << "Finished processing " << filename1 << " and " << filename2 << std::endl;
    return;
}

void rectify_images(Mat img1, Mat img2, std::vector<KeyPoint> keypoints1, std::vector<KeyPoint> keypoints2, std::vector<DMatch> correspondences, Mat fundamental_matrix){
    std::vector<Point2f> points1, points2;
    for(int i = 0; i < correspondences.size(); i++){
        points1.push_back(keypoints1[correspondences[i].queryIdx].pt);
        points2.push_back(keypoints2[correspondences[i].trainIdx].pt);   
    }
    if(points1.size() != points2.size() || points1.size() < 8){
        std::cout << "Error: The number of points is not enough." << std::endl;
        return;
    }
    Mat homography1, homography2;
    double threshold = 5.0;
    // TODO: the following function doesn't require intrinsic paramters, but we have intrinsic parameters to be used
    if(!stereoRectifyUncalibrated(points1, points2, fundamental_matrix, img1.size(), homography1, homography2, threshold)){
        std::cout << "Error: Failed to rectify images." << std::endl;
        return;
    }
    Mat rectified1, rectified2;
    // TODO: rectify images
    warpPerspective(img1, rectified1, homography1, img1.size());
    warpPerspective(img2, rectified2, homography2, img2.size());
    return;
}

Mat find_fundamental_matrix(std::vector<KeyPoint> keypoints1, std::vector<KeyPoint> keypoints2, std::vector<DMatch> correspondences){
    std::vector<Point2f> points1, points2;
    for(int i = 0; i < correspondences.size(); i++){
        points1.push_back(keypoints1[correspondences[i].queryIdx].pt);
        points2.push_back(keypoints2[correspondences[i].trainIdx].pt);   
    }
    if(points1.size() != points2.size() || points1.size() < 8){
        std::cout << "Error: The number of points is not enough." << std::endl;
        return Mat();
    }
    Mat fundamental_matrix;
    double ransacReprojThreshold = 3.0, confidence = 0.99;
    if(FUNDAMENTAL_MATRIX_METHOD == "ransac")
        fundamental_matrix = findFundamentalMat(points1, points2, FM_RANSAC, ransacReprojThreshold, confidence);
    else if(FUNDAMENTAL_MATRIX_METHOD == "lmeds")
        fundamental_matrix = findFundamentalMat(points1, points2, FM_LMEDS, ransacReprojThreshold, confidence);
    else if(FUNDAMENTAL_MATRIX_METHOD == "7point")
        fundamental_matrix = findFundamentalMat(points1, points2, FM_7POINT);
    else if(FUNDAMENTAL_MATRIX_METHOD == "8point")
        fundamental_matrix = findFundamentalMat(points1, points2, FM_8POINT);
    else{
        std::cout << "Error: No such fundamental matrix method." << std::endl;
        return Mat();
    }
    return fundamental_matrix;
}

std::vector<DMatch> match_descriptors(Mat img1, Mat img2, struct detectResult result1, struct detectResult result2){
    Mat descriptors1 = result1.descriptors, descriptors2 = result2.descriptors;
    std::vector<KeyPoint> keypoints1 = result1.keypoints, keypoints2 = result2.keypoints;
    std::vector<std::vector<DMatch>> correspondences;
    if(DESCRIPTOR_MATCHING_METHOD == "flann"){
        Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
        // the knn use NORM_L2 because sift is a float-pointing descriptor
        // descriptor1 = queryDescriptor, descriptor2 = trainDescriptor
        matcher->knnMatch(descriptors1, descriptors2, correspondences, 2);
    }
    else if(DESCRIPTOR_MATCHING_METHOD == "brute_force"){
        Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);
        // the knn use NORM_L2 because sift is a float-pointing descriptor
        matcher->knnMatch(descriptors1, descriptors2, correspondences, 2);
    }
    else{
        std::cout << "Error: No such matching method." << std::endl;
        exit(-1);
    }
    const float ratio_thresh = 0.7f;
    std::vector<DMatch> good_matches;
    for (size_t i = 0; i < correspondences.size(); i++){
        if (correspondences[i][0].distance < ratio_thresh * correspondences[i][1].distance)
            good_matches.push_back(correspondences[i][0]);
    }
    Mat img_matches;
    drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches, Scalar::all(-1), Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    imwrite("/Users/k/Desktop/courses/3dscanning/3DScanning/descriptor_match/" + std::to_string(result1.filetype.number) + "-" + std::to_string(result2.filetype.number) + ".png", img_matches);
    return good_matches;
}

// TESTED
struct detectResult detect_keypoints_or_features(std::string dataset_dir, std::string img_name, Mat img){
    if(DESCRIPTOR_METHOD == "harris") {
        int blocksize = 2, aperture_size = 3, thresh = 200;
        double k = 0.04;
        Mat img_gray, distance_norm, distance_norm_scaled;
        cvtColor(img, img_gray, COLOR_BGR2GRAY);
        Mat distance = Mat::zeros(img.size(), CV_32FC1);
        cornerHarris(img_gray, distance, blocksize, aperture_size, k);
        normalize(distance, distance_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
        convertScaleAbs(distance_norm, distance_norm_scaled);
        // namedWindow("Harris Corner", WINDOW_AUTOSIZE);
        // imshow("Harris Corner", distance_norm_scaled);
        std::vector<KeyPoint> keypoints;
        for(int i = 0; i < distance_norm.rows ; i++ ){
            for(int j = 0; j < distance_norm.cols; j++){
                if((int)distance_norm.at<float>(i,j) > thresh){
                    // TODO: how to decide the size of keypoints?
                    keypoints.push_back(KeyPoint(j*1.0, i*1.0, 2.0));
                    circle(distance_norm_scaled, Point(j,i), 5, Scalar(0), 2, 8, 0);
                }
            }
        }
        // Mat keypoints_on_image;
        // drawKeypoints(img, keypoints, keypoints_on_image, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
        // imwrite("/Users/k/Desktop/courses/3dscanning/3DScanning/harris_corner_keypoints/" + img_name, keypoints_on_image);
        // imwrite("/Users/k/Desktop/courses/3dscanning/3DScanning/harris_corner/" + img_name, distance_norm_scaled);
        // imwrite("/Users/k/Desktop/courses/3dscanning/3DScanning/harris_corner_unlabeled/" + img_name, distance_norm);
        // https://docs.opencv.org/3.4/d4/d7d/tutorial_harris_detector.html
        struct detectResult result;
        result.keypoints = keypoints;
        result.descriptors = distance_norm_scaled; // TODO: how to get descriptors?
        result.filetype = extract_file_name(img_name);
        return result;
    }
    else if(DESCRIPTOR_METHOD == "sift") {
        // std::cout << "detecting sift keypoints and descriptors" << std::endl;
        int nfeatures = 0, nOctaveLayers = 3;
        double contrastThreshold = 0.04, edgeThreshold = 10, sigma = 1.6;
        Ptr<SIFT> sift_detector = SIFT::create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
        std::vector<KeyPoint> keypoints;
        Mat descriptors;
        sift_detector->detectAndCompute(img, Mat(), keypoints, descriptors);
        // Mat keypoints_on_image;
        // drawKeypoints(img, keypoints, keypoints_on_image, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
        // imwrite("/Users/k/Desktop/courses/3dscanning/3DScanning/sift/" + img_name, keypoints_on_image);
        // https://docs.opencv.org/3.4/d7/d66/tutorial_feature_detection.html 
        struct detectResult result;
        result.keypoints = keypoints;
        result.descriptors = descriptors;
        result.filetype = extract_file_name(img_name);
        return result;
    }
}

// TESTED
int main()
{
    std::string dataset_dir = "/Users/k/Desktop/courses/3dscanning/3DScanning/bricks-rgbd/";
    DIR* directory = opendir(dataset_dir.c_str());
    struct dirent* entry;
    int count = 0;
    if(directory == NULL){
        std::cout << "Error: Directory not found or failed to open directory." << std::endl;
        return -1;
    } else {
        while((entry = readdir(directory)) != NULL){
            if(entry->d_name[0] != 'f')
                continue;
            count++;
            struct filenameType filename_type = extract_file_name(entry->d_name);
            if(filename_type.category != 0)
                continue;
            process_pair_images(dataset_dir, entry->d_name, get_file_name(filename_type.number+1, filename_type.category));
        }
        closedir(directory);
    }
    std::cout << "Stereo Reconstruction Finished" << count  << std::endl;
    return 0;
}
