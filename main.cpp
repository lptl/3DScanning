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

#include "Libs/Eigen.h"
#include "Libs/VirtualSensor.h"
#include "Libs/Types.h"

#define METHOD "harris"

using namespace cv;
using namespace cv::xfeatures2d;

void detect_keypoints_or_features(Mat);  
struct filenameType extract_file_name(std::string);
std::string get_file_name(int, int);
void process_pair_images(std::string, std::string, std::string);
std::vector<KeyPoint> detect_keypoints_or_features(std::string, std::string, Mat);

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

std::string get_file_name(int number, int category){
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
    else
        filename = "";
    return filename;
}

void process_pair_images(std::string dataset_dir, std::string filename1, std::string filename2){
    Mat img1 = imread(dataset_dir + filename1, IMREAD_COLOR);
    Mat img2 = imread(dataset_dir + filename2, IMREAD_COLOR);
    if(img1.empty() || img2.empty()){
        std::cout << "Error: Image not found or failed to open image." << std::endl;
        return;
    }
    detect_keypoints_or_features(dataset_dir, filename1, img1);
    detect_keypoints_or_features(dataset_dir, filename1, img2);
    // correspondence_match();
}

std::vector<KeyPoint> detect_keypoints_or_features(std::string dataset_dir, std::string img_name, Mat img){
    if(METHOD == "harris") {
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
        // TODO: extract keypoints from distance_norm_scaled
        // https://docs.opencv.org/3.4/d4/d7d/tutorial_harris_detector.html
        // return distance_norm_scaled;
        return keypoints;
    }
    else if(METHOD == "sift") {
        // std::cout << "detecting sift keypoints and descriptors" << std::endl;
        int nfeatures = 0, nOctaveLayers = 3;
        double contrastThreshold = 0.04, edgeThreshold = 10, sigma = 1.6;
        Ptr<SIFT> sift_detector = SIFT::create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
        std::vector<KeyPoint> keypoints;
        Mat descriptors;
        sift_detector->detectAndCompute(img, Mat(), keypoints, descriptors);
        // std::cout << keypoints[0].size << std::endl;
        // TODO: what should be returned? keypoints or descriptors or both?
        // Mat keypoints_on_image;
        // drawKeypoints(img, keypoints, keypoints_on_image, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
        // imwrite("/Users/k/Desktop/courses/3dscanning/3DScanning/sift/" + img_name, keypoints_on_image);
        // https://docs.opencv.org/3.4/d7/d66/tutorial_feature_detection.html 
        return keypoints;
    }
}

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
