#pragma once

#include <iostream>
#include <fstream>
#include <array>

#include "OpenCVLib.h"
#include "Utils.h"

#define DESCRIPTOR_METHOD "brisk" // harris, sift, surf, orb, brisk
#define DESCRIPTOR_MATCHING_METHOD "brute_force" // flann, brute_force
#define FUNDAMENTAL_MATRIX_METHOD "ransac" // ransac, lmeds, 7point, 8point
#define MATCHING_METHOD "bm" // bm, sgbm

using namespace cv;
using namespace cv::xfeatures2d;
using namespace cv::ximgproc;

std::string PROJECT_PATH = "E:/Study/Courses/3DScanningMotionCapture/Project/";

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
    if(compare_string(FUNDAMENTAL_MATRIX_METHOD, "ransac"))
        fundamental_matrix = findFundamentalMat(points1, points2, FM_RANSAC, ransacReprojThreshold, confidence);
    else if(compare_string(FUNDAMENTAL_MATRIX_METHOD, "lmeds"))
        fundamental_matrix = findFundamentalMat(points1, points2, FM_LMEDS, ransacReprojThreshold, confidence);
    else if(compare_string(FUNDAMENTAL_MATRIX_METHOD, "7point"))
        fundamental_matrix = findFundamentalMat(points1, points2, FM_7POINT);
    else if(compare_string(FUNDAMENTAL_MATRIX_METHOD, "8point"))
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
    if(compare_string(DESCRIPTOR_MATCHING_METHOD, "flann")){
        Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
        // the knn use NORM_L2 because sift is a float-pointing descriptor
        // descriptor1 = queryDescriptor, descriptor2 = trainDescriptor
        std::cout << "peforming flann based matching" << std::endl;
        matcher->knnMatch(descriptors1, descriptors2, correspondences, 2);
    }
    else if(compare_string(DESCRIPTOR_MATCHING_METHOD, "brute_force")){
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
    imwrite(PROJECT_PATH + "descriptor_match/" + std::to_string(result1.filetype.number) + "-" + std::to_string(result2.filetype.number) + ".png", img_matches);
    return good_matches;
}

struct detectResult detect_keypoints_or_features(std::string dataset_dir, std::string img_name, Mat img){
    if(compare_string(DESCRIPTOR_METHOD, "harris")) {
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
        // imwrite(PROJECT_PATH + "harris_corner_unlabeled/" + img_name, distance_norm);
        // https://docs.opencv.org/3.4/d4/d7d/tutorial_harris_detector.html
        struct detectResult result;
        result.keypoints = keypoints;
        int nfeatures = 0, nOctaveLayers = 3;
        double contrastThreshold = 0.04, edgeThreshold = 10, sigma = 1.6;
        Ptr<SIFT> sift_detector = SIFT::create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
        Mat descriptors;
        sift_detector->compute(img, keypoints, descriptors);
        result.descriptors = descriptors; // TODO: use the sift descriptor for every keypoint detecting method
        result.filetype = extract_file_name(img_name);
        return result;
    }
    else if(compare_string(DESCRIPTOR_METHOD, "sift")) {
        int nfeatures = 0, nOctaveLayers = 3;
        double contrastThreshold = 0.04, edgeThreshold = 10, sigma = 1.6;
        Ptr<SIFT> sift_detector = SIFT::create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
        std::vector<KeyPoint> keypoints;
        Mat descriptors;
        sift_detector->detectAndCompute(img, Mat(), keypoints, descriptors);
        // Mat keypoints_on_image;
        // drawKeypoints(img, keypoints, keypoints_on_image, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
        // imwrite(PROJECT_PATH + "sift/" + img_name, keypoints_on_image);
        // https://docs.opencv.org/3.4/d7/d66/tutorial_feature_detection.html 
        struct detectResult result;
        result.keypoints = keypoints;
        result.descriptors = descriptors;
        result.filetype = extract_file_name(img_name);
        return result;
    }
    else if(compare_string(DESCRIPTOR_METHOD, "surf")) {
        // std::cout << "detecting surf keypoints and descriptors" << std::endl;
        int minHessian = 400;
        Ptr<SURF> surf_detector = SURF::create(minHessian);
        std::vector<KeyPoint> keypoints;
        Mat descriptors;
        surf_detector->detectAndCompute(img, Mat(), keypoints, descriptors);
        // Mat keypoints_on_image;
        // drawKeypoints(img, keypoints, keypoints_on_image, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
        // imwrite(PROJECT_PATH + "surf/" + img_name, keypoints_on_image);
        // https://docs.opencv.org/3.4/d7/d66/tutorial_feature_detection.html 
        struct detectResult result;
        result.keypoints = keypoints;
        result.descriptors = descriptors;
        result.filetype = extract_file_name(img_name);
        return result;
    }
    else if(compare_string(DESCRIPTOR_METHOD, "orb")) {
        // std::cout << "detecting orb keypoints and descriptors" << std::endl;
        int nfeatures = 500;
        float scaleFactor = 1.2f;
        int nlevels = 8;
        int edgeThreshold = 31;
        int firstLevel = 0;
        int WTA_K = 2;
        int patchSize = 31;
        int fastThreshold = 20;
        Ptr<ORB> orb_detector = ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K, ORB::HARRIS_SCORE, patchSize, fastThreshold);
        std::vector<KeyPoint> keypoints;
        Mat descriptors;
        orb_detector->detectAndCompute(img, Mat(), keypoints, descriptors);
        // Mat keypoints_on_image;
        // drawKeypoints(img, keypoints, keypoints_on_image, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
        // imwrite(PROJECT_PATH + "orb/" + img_name, keypoints_on_image);
        // https://docs.opencv.org/3.4/d7/d66/tutorial_feature_detection.html 
        struct detectResult result;
        result.keypoints = keypoints;
        result.descriptors = descriptors;
        result.filetype = extract_file_name(img_name);
        return result;
    }
    else if(compare_string(DESCRIPTOR_METHOD, "brisk")) {
        int thresh = 30;
        int octaves = 3;
        float patternScale = 1.0f;
        Ptr<BRISK> brisk_detector = BRISK::create(thresh, octaves, patternScale);
        std::vector<KeyPoint> keypoints;
        Mat descriptors;
        brisk_detector->detectAndCompute(img, Mat(), keypoints, descriptors);
        // Mat keypoints_on_image;
        // drawKeypoints(img, keypoints, keypoints_on_image, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
        // imwrite(PROJECT_PATH + "brisk/" + img_name, keypoints_on_image);
        // https://docs.opencv.org/3.4/d7/d66/tutorial_feature_detection.html 
        struct detectResult result;
        result.keypoints = keypoints;
        result.descriptors = descriptors;
        result.filetype = extract_file_name(img_name);
        return result;
    }
    else if (compare_string(DESCRIPTOR_METHOD, "shi-tomasi")) {
        std::vector<Point2f> corners;
        double qualityLevel = 0.01;
        double minDistance = 10;
        int maxCorners = 0, blockSize = 3, gradientSize = 3;
        bool useHarrisDetector = false;
        double k = 0.04;
        Mat img_gray;
        cvtColor(img, img_gray, COLOR_BGR2GRAY);

        goodFeaturesToTrack(img_gray,
                            corners,
                            maxCorners,
                            qualityLevel,
                            minDistance,
                            Mat(),
                            blockSize,
                            gradientSize,
                            useHarrisDetector,
                            k);

        std::vector<KeyPoint> keypoints;
        for (size_t i = 0; i < corners.size(); i++) {
            keypoints.push_back(KeyPoint(corners[i].x, corners[i].y, 4.0));
        }

        /*
        Mat image_with_keypoints;
        drawKeypoints(img, keypoints, image_with_keypoints, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
        imshow("Shi-Tomasi", image_with_keypoints);
        waitKey();
        */

        int nfeatures = 0, nOctaveLayers = 3;
        double contrastThreshold = 0.04, edgeThreshold = 10, sigma = 1.6;
        Ptr<SIFT> siftDetector = SIFT::create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
        Mat descriptors;
        siftDetector->compute(img, keypoints, descriptors);
        
        struct detectResult result;
        result.keypoints = keypoints;
        result.descriptors = descriptors;
        result.filetype = extract_file_name(img_name);
        return result;
    }
    else if (compare_string(DESCRIPTOR_METHOD, "fast")) {
        Ptr<FastFeatureDetector> fastDetector = FastFeatureDetector::create();
        
        Mat img_gray;
        cvtColor(img, img_gray, COLOR_BGR2GRAY);
        std::vector<KeyPoint> keypoints;

        fastDetector->detect(img_gray, keypoints);

        /*
        Mat image_with_keypoints;
        drawKeypoints(img, keypoints, image_with_keypoints, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
        imshow("FAST", image_with_keypoints);
        waitKey();
        */

        int nfeatures = 0, nOctaveLayers = 3;
        double contrastThreshold = 0.04, edgeThreshold = 10, sigma = 1.6;
        Ptr<SIFT> siftDetector = SIFT::create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
        Mat descriptors;
        siftDetector->compute(img, keypoints, descriptors);
        
        struct detectResult result;
        result.keypoints = keypoints;
        result.descriptors = descriptors;
        result.filetype = extract_file_name(img_name);
        return result;
    }
    struct detectResult result;
    return result;
}

Mat compute_disparity_map(Mat left, Mat right) {
    if (compare_string(MATCHING_METHOD, "bm")) {
        int max_disp = 160, wsize = 15;
        Ptr<StereoBM> left_matcher = StereoBM::create(max_disp, wsize);

        Mat left_gray, right_gray, left_disp;
        cvtColor(left, left_gray, COLOR_BGR2GRAY);
        cvtColor(right, right_gray, COLOR_BGR2GRAY);

        left_matcher->compute(left_gray, right_gray, left_disp);

        /*
        Mat left_disp_vis;
        getDisparityVis(left_disp, left_disp_vis);
        imshow("Left Disparity", left_disp_vis);
        waitKey();
        */

        return left_disp;
    }
    else if (compare_string(MATCHING_METHOD, "sgbm")) {
        int max_disp = 160, wsize = 3;
        Ptr<StereoSGBM> left_matcher = StereoSGBM::create(0, max_disp, wsize);
        left_matcher->setP1(24 * wsize * wsize);
        left_matcher->setP2(96 * wsize * wsize);
        left_matcher->setPreFilterCap(63);
        left_matcher->setMode(StereoSGBM::MODE_SGBM_3WAY);

        Mat left_gray, right_gray, left_disp;
        cvtColor(left, left_gray, COLOR_BGR2GRAY);
        cvtColor(right, right_gray, COLOR_BGR2GRAY);

        left_matcher->compute(left_gray, right_gray, left_disp);

        /*
        Mat left_disp_vis;
        getDisparityVis(left_disp, left_disp_vis);
        imshow("Left Disparity", left_disp_vis);
        waitKey();
        */

        return left_disp;
    }
}