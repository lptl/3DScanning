#pragma once

#include <iostream>
#include <array>

#include "SimpleMesh.h"
#include "PointCloud.h"
#include "ProcrustesAligner.h"
#include "NearestNeighbour.h"
#include "ICPOptimizer.h"

#include "Utils.h"

#define DESCRIPTOR_METHOD "brisk" // harris, sift, surf, orb, brisk, shi-tomasi, fast
#define DESCRIPTOR_MATCHING_METHOD "flann" // flann, brute_force
#define FUNDAMENTAL_MATRIX_METHOD "ransac" // ransac, lmeds, 7point, 8point
#define DENSE_MATCHING_METHOD "sgbm" // bm, sgbm
#define USE_POST_FILTERING true
#define USE_LINEAR_ICP true
#define USE_POINT_TO_PLANE false

std::string MODELS_DIR = "Models/";
std::string PROJECT_PATH = "../../";



void rectify_images(cv::Mat img1, cv::Mat img2, std::vector<cv::KeyPoint> keypoints1, std::vector<cv::KeyPoint> keypoints2, std::vector<cv::DMatch> good_matches, cv::Mat fundamental_matrix, struct cameraParams camParams, cv::Mat& left_rectified, cv::Mat& right_rectified) {
    
    if (!camParams.empty) {
        cv::Mat R1, R2, P1, P2, Q;
        // TODO: Currently this should give the ground-truth rectified images. For rectified images based on our method, decompose the fundamental matrix into R and T and pass those here (instead of left_to_right_R and left_to_right_T respectively)
        stereoRectify(camParams.left_camera_matrix, camParams.left_distortion_coeffs, camParams.right_camera_matrix, camParams.right_distortion_coeffs, img1.size(), camParams.left_to_right_R, camParams.left_to_right_T, R1, R2, P1, P2, Q);

        cv::Mat rmap[2][2];
        initUndistortRectifyMap(camParams.left_camera_matrix, camParams.left_distortion_coeffs, R1, P1, img1.size(), CV_16SC2, rmap[0][0], rmap[0][1]);
        initUndistortRectifyMap(camParams.right_camera_matrix, camParams.right_distortion_coeffs, R2, P2, img1.size(), CV_16SC2, rmap[1][0], rmap[1][1]);

        remap(img1, left_rectified, rmap[0][0], rmap[0][1], cv::INTER_LINEAR);
        remap(img2, right_rectified, rmap[1][0], rmap[1][1], cv::INTER_LINEAR);
    }
    else {
        std::vector<cv::Point2f> points1, points2;
        for (int i = 0; i < good_matches.size(); i++) {
            points1.push_back(keypoints1[good_matches[i].queryIdx].pt);
            points2.push_back(keypoints2[good_matches[i].trainIdx].pt);
        }
        if (points1.size() != points2.size() || points1.size() < 8) {
            std::cout << "Error: The number of points is not enough." << std::endl;
            return;
        }

        cv::Mat homography1, homography2;
        double threshold = 5.0;
        if (!stereoRectifyUncalibrated(points1, points2, fundamental_matrix, img1.size(), homography1, homography2, threshold)) {
            std::cout << "Error: Failed to rectify images." << std::endl;
            return;
        }
        
        warpPerspective(img1, left_rectified, homography1, img1.size());
        warpPerspective(img2, right_rectified, homography2, img2.size());
    }
    
    return;
}

void find_fundamental_matrix(std::vector<cv::KeyPoint> keypoints1, std::vector<cv::KeyPoint> keypoints2, std::vector<cv::DMatch> correspondences, cv::Mat& fundamental_matrix){
    std::vector<cv::Point2f> points1, points2;
    for(int i = 0; i < correspondences.size(); i++){
        points1.push_back(keypoints1[correspondences[i].queryIdx].pt);
        points2.push_back(keypoints2[correspondences[i].trainIdx].pt);   
    }
    if(points1.size() != points2.size() || points1.size() < 8){
        std::cout << "Error: The number of points is not enough." << std::endl;
        return;
    }

    double ransacReprojThreshold = 3.0, confidence = 0.99;
    if(compare_string(FUNDAMENTAL_MATRIX_METHOD, "ransac"))
        // Need 15 points for RANSAC, maybe need to change the condition above
        fundamental_matrix = findFundamentalMat(points1, points2, cv::FM_RANSAC, ransacReprojThreshold, confidence);
    else if(compare_string(FUNDAMENTAL_MATRIX_METHOD, "lmeds"))
        fundamental_matrix = findFundamentalMat(points1, points2, cv::FM_LMEDS, ransacReprojThreshold, confidence);
    else if(compare_string(FUNDAMENTAL_MATRIX_METHOD, "7point"))
        fundamental_matrix = findFundamentalMat(points1, points2, cv::FM_7POINT);
    else if(compare_string(FUNDAMENTAL_MATRIX_METHOD, "8point"))
        fundamental_matrix = findFundamentalMat(points1, points2, cv::FM_8POINT);
    else{
        std::cout << "Error: No such fundamental matrix method." << std::endl;
        return;
    }
    return;
}

void match_descriptors(cv::Mat descriptors1, cv::Mat descriptors2, std::vector<cv::DMatch>& good_matches) {
    std::vector<std::vector<cv::DMatch>> correspondences;

    if(compare_string(DESCRIPTOR_MATCHING_METHOD, "flann")){
        cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
        // the knn use NORM_L2 because sift is a float-pointing descriptor
        // descriptor1 = queryDescriptor, descriptor2 = trainDescriptor
        descriptors1.convertTo(descriptors1, CV_32F);
        descriptors2.convertTo(descriptors2, CV_32F);
        matcher->knnMatch(descriptors1, descriptors2, correspondences, 2);
    }
    else if(compare_string(DESCRIPTOR_MATCHING_METHOD, "brute_force")){
        cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE);
        // the knn use NORM_L2 because sift is a float-pointing descriptor
        matcher->knnMatch(descriptors1, descriptors2, correspondences, 2);
    }
    else{
        std::cout << "Error: No such matching method." << std::endl;
        exit(-1);
    }

    const float ratio_thresh = 0.7f;
    for (size_t i = 0; i < correspondences.size(); i++){
        if (correspondences[i][0].distance < ratio_thresh * correspondences[i][1].distance)
            good_matches.push_back(correspondences[i][0]);
    }
    
    return;
}

void detect_keypoints_or_features(std::string img_name, cv::Mat img, struct detectResult *result){

    if(compare_string(DESCRIPTOR_METHOD, "harris")) {
        int blocksize = 2, aperture_size = 3, thresh = 200;
        double k = 0.04;
        cv::Mat img_gray, distance_norm, distance_norm_scaled;
        cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
        cv::Mat distance = cv::Mat::zeros(img.size(), CV_32FC1);
        cornerHarris(img_gray, distance, blocksize, aperture_size, k);
        normalize(distance, distance_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
        convertScaleAbs(distance_norm, distance_norm_scaled);
        // namedWindow("Harris Corner", WINDOW_AUTOSIZE);
        // imshow("Harris Corner", distance_norm_scaled);
        std::vector<cv::KeyPoint> keypoints;
        for(int i = 0; i < distance_norm.rows ; i++ ){
            for(int j = 0; j < distance_norm.cols; j++){
                if((int)distance_norm.at<float>(i,j) > thresh){
                    // TODO: how to decide the size of keypoints?
                    keypoints.push_back(cv::KeyPoint(j*1.0, i*1.0, 2.0));
                    circle(distance_norm_scaled, cv::Point(j,i), 5, cv::Scalar(0), 2, 8, 0);
                }
            }
        }
        // cv::Mat keypoints_on_image;
        // drawKeypoints(img, keypoints, keypoints_on_image, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
        // imwrite(PROJECT_PATH + "harris_corner_unlabeled/" + img_name, distance_norm);
        // https://docs.opencv.org/3.4/d4/d7d/tutorial_harris_detector.html

        result->keypoints = keypoints;
        int nfeatures = 0, nOctaveLayers = 3;
        double contrastThreshold = 0.04, edgeThreshold = 10, sigma = 1.6;
        cv::Ptr<cv::SIFT> sift_detector = cv::SIFT::create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
        cv::Mat descriptors;
        sift_detector->compute(img, keypoints, descriptors);
        result->descriptors = descriptors; // TODO: use the sift descriptor for every keypoint detecting method
        result->filetype = extract_file_name(img_name);
    }
    else if(compare_string(DESCRIPTOR_METHOD, "sift")) {
        int nfeatures = 0, nOctaveLayers = 3;
        double contrastThreshold = 0.04, edgeThreshold = 10, sigma = 1.6;
        cv::Ptr<cv::SIFT> sift_detector = cv::SIFT::create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        sift_detector->detectAndCompute(img, cv::Mat(), keypoints, descriptors);
        // cv::Mat keypoints_on_image;
        // drawKeypoints(img, keypoints, keypoints_on_image, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
        // imwrite(PROJECT_PATH + "sift/" + img_name, keypoints_on_image);
        // https://docs.opencv.org/3.4/d7/d66/tutorial_feature_detection.html 

        result->keypoints = keypoints;
        result->descriptors = descriptors;
        result->filetype = extract_file_name(img_name);
    }
    else if(compare_string(DESCRIPTOR_METHOD, "surf")) {
        // std::cout << "detecting surf keypoints and descriptors" << std::endl;
        int minHessian = 400;
        cv::Ptr<cv::xfeatures2d::SURF> surf_detector = cv::xfeatures2d::SURF::create(minHessian);
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        surf_detector->detectAndCompute(img, cv::Mat(), keypoints, descriptors);
        // cv::Mat keypoints_on_image;
        // drawKeypoints(img, keypoints, keypoints_on_image, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
        // imwrite(PROJECT_PATH + "surf/" + img_name, keypoints_on_image);
        // https://docs.opencv.org/3.4/d7/d66/tutorial_feature_detection.html 

        result->keypoints = keypoints;
        result->descriptors = descriptors;
        result->filetype = extract_file_name(img_name);
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
        cv::Ptr<cv::ORB> orb_detector = cv::ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K, cv::ORB::HARRIS_SCORE, patchSize, fastThreshold);
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        orb_detector->detectAndCompute(img, cv::Mat(), keypoints, descriptors);
        // cv::Mat keypoints_on_image;
        // drawKeypoints(img, keypoints, keypoints_on_image, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
        // imwrite(PROJECT_PATH + "orb/" + img_name, keypoints_on_image);
        // https://docs.opencv.org/3.4/d7/d66/tutorial_feature_detection.html 

        result->keypoints = keypoints;
        result->descriptors = descriptors;
        result->filetype = extract_file_name(img_name);
    }
    else if(compare_string(DESCRIPTOR_METHOD, "brisk")) {
        int thresh = 30;
        int octaves = 3;
        float patternScale = 1.0f;
        cv::Ptr<cv::BRISK> brisk_detector = cv::BRISK::create(thresh, octaves, patternScale);
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        brisk_detector->detectAndCompute(img, cv::Mat(), keypoints, descriptors);
        // cv::Mat keypoints_on_image;
        // drawKeypoints(img, keypoints, keypoints_on_image, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
        // imwrite(PROJECT_PATH + "brisk/" + img_name, keypoints_on_image);
        // https://docs.opencv.org/3.4/d7/d66/tutorial_feature_detection.html 

        result->keypoints = keypoints;
        result->descriptors = descriptors;
        result->filetype = extract_file_name(img_name);
    }
    else if (compare_string(DESCRIPTOR_METHOD, "shi-tomasi")) {
        std::vector<cv::Point2f> corners;
        double qualityLevel = 0.01;
        double minDistance = 10;
        int maxCorners = 0, blockSize = 3, gradientSize = 3;
        bool useHarrisDetector = false;
        double k = 0.04;
        cv::Mat img_gray;
        cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);

        goodFeaturesToTrack(img_gray,
                            corners,
                            maxCorners,
                            qualityLevel,
                            minDistance,
                            cv::Mat(),
                            blockSize,
                            gradientSize,
                            useHarrisDetector,
                            k);

        std::vector<cv::KeyPoint> keypoints;
        for (size_t i = 0; i < corners.size(); i++) {
            keypoints.push_back(cv::KeyPoint(corners[i].x, corners[i].y, 4.0));
        }

        /*
        cv::Mat image_with_keypoints;
        drawKeypoints(img, keypoints, image_with_keypoints, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
        imshow("Shi-Tomasi", image_with_keypoints);
        waitKey();
        */

        int nfeatures = 0, nOctaveLayers = 3;
        double contrastThreshold = 0.04, edgeThreshold = 10, sigma = 1.6;
        cv::Ptr<cv::SIFT> siftDetector = cv::SIFT::create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
        cv::Mat descriptors;
        siftDetector->compute(img, keypoints, descriptors);

        result->keypoints = keypoints;
        result->descriptors = descriptors;
        result->filetype = extract_file_name(img_name);
    }
    else if (compare_string(DESCRIPTOR_METHOD, "fast")) {
        cv::Ptr<cv::FastFeatureDetector> fastDetector = cv::FastFeatureDetector::create();
        cv::Mat img_gray;
        cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
        std::vector<cv::KeyPoint> keypoints;
        fastDetector->detect(img_gray, keypoints);

        /*
        cv::Mat image_with_keypoints;
        drawKeypoints(img, keypoints, image_with_keypoints, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
        imshow("FAST", image_with_keypoints);
        waitKey();
        */
        int nfeatures = 0, nOctaveLayers = 3;
        double contrastThreshold = 0.04, edgeThreshold = 10, sigma = 1.6;
        cv::Ptr<cv::SIFT> siftDetector = cv::SIFT::create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
        cv::Mat descriptors;
        siftDetector->compute(img, keypoints, descriptors);
        
        result->keypoints = keypoints;
        result->descriptors = descriptors;
        result->filetype = extract_file_name(img_name);
    }
    else {
        std::cout << "No such keypoint/descriptor method." << std::endl;
    }
    return;
}

void compute_disparity_map(cv::Mat left, cv::Mat right, cv::Mat& disp) {

    cv::Ptr<cv::ximgproc::DisparityWLSFilter> wls_filter;
    cv::Mat left_gray, right_gray, left_disp, right_disp;
    cvtColor(left, left_gray, cv::COLOR_BGR2GRAY);
    cvtColor(right, right_gray, cv::COLOR_BGR2GRAY);

    if (compare_string(DENSE_MATCHING_METHOD, "bm")) {
        int max_disp = 160, wsize = 15;
        cv::Ptr<cv::StereoBM> left_matcher = cv::StereoBM::create(max_disp, wsize);
        cv::Ptr<cv::StereoMatcher> right_matcher = cv::ximgproc::createRightMatcher(left_matcher);

        if (!USE_POST_FILTERING) {
            left_matcher->compute(left_gray, right_gray, disp);
            return;
        }
        else {
            left_matcher->compute(left_gray, right_gray, left_disp);
            right_matcher->compute(right_gray, left_gray, right_disp);
            wls_filter = cv::ximgproc::createDisparityWLSFilter(left_matcher);
        }

        /*
        cv::Mat left_disp_vis;
        getDisparityVis(left_disp, left_disp_vis);
        imshow("Left Disparity", left_disp_vis);
        waitKey();
        */
    }
    else if (compare_string(DENSE_MATCHING_METHOD, "sgbm")) {
        int min_disp = 0, max_disp = 160, wsize = 3;
        cv::Ptr<cv::StereoSGBM> left_matcher = cv::StereoSGBM::create(min_disp, max_disp, wsize);
        left_matcher->setP1(24 * wsize * wsize);
        left_matcher->setP2(96 * wsize * wsize);
        left_matcher->setMode(cv::StereoSGBM::MODE_SGBM);
        cv::Ptr<cv::StereoMatcher> right_matcher = cv::ximgproc::createRightMatcher(left_matcher);

        if (!USE_POST_FILTERING) {
            left_matcher->compute(left_gray, right_gray, disp);
            return;
        }
        else {
            left_matcher->compute(left_gray, right_gray, left_disp);
            right_matcher->compute(right_gray, left_gray, right_disp);
            wls_filter = cv::ximgproc::createDisparityWLSFilter(left_matcher);
        }

    }
    else {
        std::cout << "No such dense matching method." << std::endl;
        return;
    }

    double lambda = 8000.0, sigma = 1.5;
    wls_filter->setLambda(lambda);
    wls_filter->setSigmaColor(sigma);

    wls_filter->filter(left_disp, left, disp, right_disp);

    return;
}

void get_depth_map_from_disparity_map(cv::Mat disparityMap, struct cameraParams camParams, cv::Mat& depthMap) {

    for (int i = 0; i < disparityMap.rows; i++) {
        for (int j = 0; j < disparityMap.cols; j++) {
            double d = static_cast<double>(disparityMap.at<float>(i, j));
            depthMap.at<unsigned short>(i, j) = (camParams.baseline * camParams.fX) / d;
            depthMap.at<unsigned short>(i, j) = d;
            if (d < 10)
                depthMap.at<unsigned short>(i, j) = -1;
            //if (d>0) std::cout << "The disparity is: "<< (camParams.baseline * camParams.fX) / d<< std::endl;
            short disparity_ij = disparityMap.at<unsigned short>(i, j);
            if (disparity_ij <= 1)
                depthMap.at<unsigned short>(i, j) = 0;

            if (std::isnan(static_cast<double>(depthMap.at<unsigned short>(i, j))) || std::isinf(static_cast<double>(depthMap.at<unsigned short>(i, j))))
                depthMap.at<unsigned short>(i, j) = 0;
        }
    }

    return;
}

void get_point_cloud_from_depth_map(cv::Mat depth_map, cv::Mat rgb_map, struct cameraParams camParams, std::string filename) {

    int width = depth_map.cols;
    int height = depth_map.rows;

    Vertex* vertices = new Vertex[width * height];
    for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
            int idx = h * width + w;
            // float depth = *(depthMap + idx);
            float depth = (float)(depth_map.at<short>(h, w));
            depth = depth;
            if (depth != MINF && depth != 0 && depth < 100) { // range filter: (0, 1 meter)
                float X_c = (float(w) - camParams.cX) * depth / camParams.fX;
                float Y_c = (float(h) - camParams.cY) * depth / camParams.fY;
                Vector4f P_c = Vector4f(X_c, Y_c, depth, 1);

                vertices[idx].position = P_c;
                unsigned char R = rgb_map.at<cv::Vec3b>(h,w)[2];
                unsigned char G = rgb_map.at<cv::Vec3b>(h,w)[1];
                unsigned char B = rgb_map.at<cv::Vec3b>(h,w)[0];
                unsigned char A = 255;
                vertices[idx].color = Vector4uc(R, G, B, A);
            }
            else {
                vertices[idx].position = Vector4f(MINF, MINF, MINF, MINF);
                vertices[idx].color = Vector4uc(0, 0, 0, 0);
            }
        }
    }
    
    // write to the off file
    std::stringstream ss;
    ss << PROJECT_PATH << "Output/pointclouds/" << filename << ".off";
    writeMesh(vertices, width, height, ss.str());
    
    return;
}

bool icp_reconstruct(const std::string base_model, const std::string other_model, std::string& target_model) {
    SimpleMesh sourceMesh;
    if (!sourceMesh.loadMesh(base_model)) {
        std::cout << "Mesh file wasn't read successfully at location: " << base_model << std::endl;
        return false;
    }

    SimpleMesh targetMesh;
    if (!targetMesh.loadMesh(other_model)) {
        std::cout << "Mesh file wasn't read successfully at location: " << other_model << std::endl;
        return false;
    }

    // Estimate the pose from source to target mesh with ICP optimization.
    ICPOptimizer* optimizer = nullptr;
    if (USE_LINEAR_ICP) {
        std::cout << "USING LINEAR ICP" << std::endl;
        optimizer = new LinearICPOptimizer();
    }
    else {
        optimizer = new CeresICPOptimizer();
    }

    optimizer->setMatchingMaxDistance(0.0003f);
    if (USE_POINT_TO_PLANE) {
        std::cout << "USING POINT TO PLANE" << std::endl;
        optimizer->usePointToPlaneConstraints(true);
        optimizer->setNbOfIterations(10);
    }
    else {
        optimizer->usePointToPlaneConstraints(false);
        optimizer->setNbOfIterations(20);
    }

    PointCloud source{ sourceMesh };
    PointCloud target{ targetMesh };

    Matrix4f estimatedPose = Matrix4f::Identity();
    optimizer->estimatePose(source, target, estimatedPose);

    // Visualize the resulting joined mesh. We add triangulated spheres for point matches.
    std::cout << "joining meshes" << std::endl;
    SimpleMesh resultingMesh = SimpleMesh::joinMeshes(sourceMesh, targetMesh, estimatedPose);
    std::cout << "mesh joined" << std::endl;
    resultingMesh.writeMesh(target_model);
    std::cout << "Resulting mesh written." << std::endl;

    delete optimizer;

    return true;
}

void reconstruct(std::string models_directory) {
    DIR* directory = opendir(models_directory.c_str());
    struct dirent* entry;
    int index = 0;
    std::string base_model, target_model, other_model;
    while (directory != NULL) {
        while ((entry = readdir(directory)) != NULL) {
            if (index == 0) {
                base_model = models_directory + std::string(entry->d_name);
                index++;
                continue;
            }
            other_model = models_directory + std::string(entry->d_name);
            target_model = models_directory + std::to_string(index) + ".off";
            if (!icp_reconstruct(base_model, other_model, target_model)) {
                std::cout << "Error: Failed to reconstruct model. Skipped one model: " + base_model << std::endl;
                base_model = other_model;
            }
            else base_model = target_model;
            index++;
        }
    }
}