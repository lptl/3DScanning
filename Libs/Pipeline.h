#pragma once

//internal libs
#include <iostream>
#include <fstream>
#include <array>

#include <dirent.h>
// libs that don't involve opencv
#include "SimpleMesh.h"
#include "PointCloud.h"
#include "ProcrustesAligner.h"
#include "NearestNeighbour.h"
#include "ICPOptimizer.h"
// libs that involve opencv
#include "OpenCVLib.h"
#include "Utils.h"

#define DESCRIPTOR_METHOD "brisk" // harris, sift, surf, orb, brisk
#define DESCRIPTOR_MATCHING_METHOD "brute_force" // flann, brute_force
#define FUNDAMENTAL_MATRIX_METHOD "ransac" // ransac, lmeds, 7point, 8point

#define USE_LINEAR_ICP true
#define USE_POINT_TO_PLANE false
#define MATCHING_METHOD "bm" // bm, sgbm


using namespace cv;
using namespace cv::xfeatures2d;
using namespace cv::ximgproc;

std::string PROJECT_PATH = "/Users/k/Desktop/courses/3dscanning/3DScanning/";

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
    // TODO: rectify images from caculate homography
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

bool icp_reconstruct(const std::string base_model, const std::string other_model, std::string& target_model){
    //const std::string other_model = std::string("/Users/k/Desktop/Courses/3dscanning/3DScanning/bunny/bunny_part1.off");
    //const std::string source_model = std::string("/Users/k/Desktop/Courses/3dscanning/3DScanning/bunny/bunny_part2_trans.off");
    
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

void PointCloudGenerate(cv::Mat depth_map, int file_order = 0){
    /*
    if(dataset.name!=bricks_rgbd){
        cout<<"dataset error from pointcloud.cpp!"<<endl;
        exit(0);
    }*/
    float fX = 577.871;
    float fY = 580.258;
    float cX = 319.623;
    float cY = 239.624; 

    int width = depth_map.cols;
    int height = depth_map.rows;

    Vertex* vertices = new Vertex[width*height];
    for(int h = 0; h < height; h++){
        for(int w = 0; w < width; w++){
            int idx = h * width + w;
            // float depth = *(depthMap + idx);
            float depth = (float)(depth_map.at<short>(h,w)); 
            depth = depth;
            if(depth != MINF && depth != 0 && depth < 100){ // range filter: (0, 1 meter)
                float X_c = (float(w)-cX) * depth / fX;
                float Y_c = (float(h)-cY) * depth / fY;
                Vector4f P_c = Vector4f(X_c, Y_c, depth, 1);

                vertices[idx].position = P_c;
                /*
                unsigned char R = rgb_map.at<Vec3b>(h,w)[2];
                unsigned char G = rgb_map.at<Vec3b>(h,w)[1];
                unsigned char B = rgb_map.at<Vec3b>(h,w)[0];
                unsigned char A = 255;
                vertices[idx].color = Vector4uc(R, G, B, A); */
            }
            else{
                vertices[idx].position = Vector4f(MINF, MINF, MINF, MINF);
                vertices[idx].color = Vector4uc(0, 0, 0, 0);
            }
        }
    }
    /*
    // write to the off file
    std::stringstream ss;
    ss << "bricks_rgbd" << file_order <<".off";
    WriteMesh(vertices, width, height, ss.str());
    */
}

cv::Mat convertDisparityToDepth(const cv::Mat &disparityMap, const float baseline, const float fx) {
    cv::Mat depthMap = cv::Mat(disparityMap.size(), CV_16U);
    for (int i = 0; i < disparityMap.rows; i++) {
        for (int j = 0; j < disparityMap.cols; j++) {
            double d = static_cast<double>(disparityMap.at<float>(i, j));
            depthMap.at<unsigned short>(i, j) = (baseline * fx) / d;
            // depthMap.at<unsigned short>(i, j) = d;
            // if (d < 10)
            //     depthMap.at<unsigned short>(i, j) = -1;
            // if (d>0) std::cout << "The disparity is: "<< (baseline * fx) / d<< std::endl;
            short disparity_ij = disparityMap.at<unsigned short>(i, j);
            if(disparity_ij <= 1)
                depthMap.at<unsigned short>(i, j) = 0;

            if (std::isnan(depthMap.at<unsigned short>(i, j)) || std::isinf(depthMap.at<unsigned short>(i, j)))
                depthMap.at<unsigned short>(i, j) = 0;
        }
    }

    return depthMap;
}