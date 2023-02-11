#pragma once

#include <iostream>
#include <array>
#include <dirent.h>

#include "SimpleMesh.h"
#include "PointCloud.h"
#include "ProcrustesAligner.h"
#include "NearestNeighbour.h"
#include "ICPOptimizer.h"

#include "Utils.h"

#define DATASET "bricks-rgbd"                    // kitti, bricks-rgbd

#define DEBUG 0		      // 1, output debug information, 0, no output
#define TEST 1                  // 0: no test, 1: test 2 images only
#define USE_GROUNDTRUTH 2  // 0: no groundtruth, 1: use groundtruth disparity, 2: use groundtruth depth

#define RECONSTRUCT 0            // 0: no final model reconstruction, 1: final model reconstruction
#define COMPARE_DENSE_MATCHING 0 // 0: no comparison, 1: comparison, compare dense matching with groundtruth

#define DESCRIPTOR_METHOD "sift"            // harris, sift, surf, orb, brisk, shi-tomasi, fast
#define DESCRIPTOR_MATCHING_METHOD "flann" // flann, brute_force
#define FUNDAMENTAL_MATRIX_METHOD "ransac" // ransac, lmeds, 7point, 8point
#define DENSE_MATCHING_METHOD "sgbm"       // bm, sgbm
#define USE_POST_FILTERING true
#define USE_LINEAR_ICP true
#define USE_POINT_TO_PLANE false
#define RECONSTRUCT_METHOD "icp"      // icp, poisson
#define MERGE_METHOD "frame-to-frame" // frame-to-frame, frame-to-model
#define USE_REPROJECT false 		 // true, false

std::string MODELS_DIR = "Output/pointclouds/";
std::string PROJECT_PATH = "/Users/k/Desktop/Courses/3dscanning/3DScanning/";

cv::Mat Q_matrix;

void detect_keypoints_or_features(std::string img_name, cv::Mat img, struct detectResult *result)
{

    if (compare_string(DESCRIPTOR_METHOD, "harris"))
    {
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
        for (int i = 0; i < distance_norm.rows; i++)
        {
            for (int j = 0; j < distance_norm.cols; j++)
            {
                if ((int)distance_norm.at<float>(i, j) > thresh)
                {
                    // TODO: how to decide the size of keypoints?
                    keypoints.push_back(cv::KeyPoint(j * 1.0, i * 1.0, 2.0));
                    circle(distance_norm_scaled, cv::Point(j, i), 5, cv::Scalar(0), 2, 8, 0);
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
    else if (compare_string(DESCRIPTOR_METHOD, "sift"))
    {
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
    else if (compare_string(DESCRIPTOR_METHOD, "surf"))
    {
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
    else if (compare_string(DESCRIPTOR_METHOD, "orb"))
    {
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
    else if (compare_string(DESCRIPTOR_METHOD, "brisk"))
    {
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
    else if (compare_string(DESCRIPTOR_METHOD, "shi-tomasi"))
    {
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
        for (size_t i = 0; i < corners.size(); i++)
        {
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
    else if (compare_string(DESCRIPTOR_METHOD, "fast"))
    {
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
    else
    {
        std::cout << "Error: No such keypoint/descriptor method." << std::endl;
    }
    return;
}

void match_descriptors(cv::Mat descriptors1, cv::Mat descriptors2, std::vector<cv::DMatch> &good_matches)
{
    std::vector<std::vector<cv::DMatch>> correspondences;

    if (compare_string(DESCRIPTOR_MATCHING_METHOD, "flann"))
    {
        cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
        // the knn use NORM_L2 because sift is a float-pointing descriptor
        // descriptor1 = queryDescriptor, descriptor2 = trainDescriptor
        descriptors1.convertTo(descriptors1, CV_32F);
        descriptors2.convertTo(descriptors2, CV_32F);
        matcher->knnMatch(descriptors1, descriptors2, correspondences, 2);
    }
    else if (compare_string(DESCRIPTOR_MATCHING_METHOD, "brute_force"))
    {
        cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE);
        // the knn use NORM_L2 because sift is a float-pointing descriptor
        matcher->knnMatch(descriptors1, descriptors2, correspondences, 2);
    }
    else
    {
        std::cout << "Error: No such matching method." << std::endl;
        exit(-1);
    }

    // clean correspondences
    const float ratio_thresh = 0.7f;
    for (size_t i = 0; i < correspondences.size(); i++)
    {
        if (correspondences[i][0].distance < ratio_thresh * correspondences[i][1].distance)
            good_matches.push_back(correspondences[i][0]);
    }

    return;
}

void find_fundamental_matrix(std::vector<cv::KeyPoint> keypoints1, std::vector<cv::KeyPoint> keypoints2, std::vector<cv::DMatch> correspondences, cv::Mat &fundamental_matrix)
{
    std::vector<cv::Point2f> points1, points2;
    for (int i = 0; i < correspondences.size(); i++)
    {
        points1.push_back(keypoints1[correspondences[i].queryIdx].pt);
        points2.push_back(keypoints2[correspondences[i].trainIdx].pt);
    }
    if (points1.size() != points2.size() || points1.size() < 8)
    {
        std::cout << "Error: The number of points is not enough." << std::endl;
        return;
    }

    double ransacReprojThreshold = 3.0, confidence = 0.99;
    if (compare_string(FUNDAMENTAL_MATRIX_METHOD, "ransac"))
        // Need 15 points for RANSAC, maybe need to change the condition above
        fundamental_matrix = findFundamentalMat(points1, points2, cv::FM_RANSAC, ransacReprojThreshold, confidence);
    else if (compare_string(FUNDAMENTAL_MATRIX_METHOD, "lmeds"))
        fundamental_matrix = findFundamentalMat(points1, points2, cv::FM_LMEDS, ransacReprojThreshold, confidence);
    else if (compare_string(FUNDAMENTAL_MATRIX_METHOD, "7point"))
        fundamental_matrix = findFundamentalMat(points1, points2, cv::FM_7POINT);
    else if (compare_string(FUNDAMENTAL_MATRIX_METHOD, "8point"))
        fundamental_matrix = findFundamentalMat(points1, points2, cv::FM_8POINT);
    else
    {
        std::cout << "Error: No such fundamental matrix calculation method." << std::endl;
        return;
    }
    return;
}

void rectify_images(cv::Mat img1, cv::Mat img2, std::vector<cv::KeyPoint> keypoints1, std::vector<cv::KeyPoint> keypoints2, std::vector<cv::DMatch> good_matches, cv::Mat fundamental_matrix, struct cameraParams camParams, cv::Mat &left_rectified, cv::Mat &right_rectified)
{

    if (!camParams.empty)
    {
        cv::Mat R1, R2, P1, P2, Q;
        cv::Mat essential_matrix = camParams.right_camera_matrix.t() * fundamental_matrix * camParams.left_camera_matrix;
        // TODO: the calculated translation matrix has large deviation from the groundtruth one.
        cv::Mat U, S, Vt;
        cv::SVDecomp(essential_matrix, S, U, Vt, cv::SVD::FULL_UV);
        cv::Mat W = (cv::Mat_<double>(3, 3) << 0, -1, 0, 1, 0, 0, 0, 0, 1);
        cv::Mat rotation_matrix = U * W.t() * Vt;  // or U * W.t() * Vt
        cv::Mat translation_matrix = -U.col(2); // or -U.col(2)
	if(DEBUG){
	  std::cout << "Rectifying with camera parameters..." << std::endl;
	  std::cout << "calculated rotation matrix: " << std::endl
		    << rotation_matrix << std::endl;
	  std::cout << "calculated translation matrix: " << std::endl
		    << translation_matrix << std::endl;
	  std::cout << "real rotation matrix: " << std::endl
		    << camParams.right_to_left_R << std::endl;
	  std::cout << "real translation matrix: " << camParams.right_to_left_T << std::endl;
	}
        // // stereoRectify(camParams.left_camera_matrix, camParams.left_distortion_coeffs, camParams.right_camera_matrix, camParams.right_distortion_coeffs, img1.size(), rotation_matrix, translation_matrix, R1, R2, P1, P2, Q);
        stereoRectify(camParams.left_camera_matrix, camParams.left_distortion_coeffs, camParams.right_camera_matrix, camParams.right_distortion_coeffs, img1.size(), camParams.left_to_right_R, camParams.left_to_right_T, R1, R2, P1, P2, Q);
	Q_matrix = Q;
	
        cv::Mat rmap[2][2];
        initUndistortRectifyMap(camParams.left_camera_matrix, camParams.left_distortion_coeffs, R1, P1, img1.size(), CV_16SC2, rmap[0][0], rmap[0][1]);
        initUndistortRectifyMap(camParams.right_camera_matrix, camParams.right_distortion_coeffs, R2, P2, img1.size(), CV_16SC2, rmap[1][0], rmap[1][1]);

        remap(img1, left_rectified, rmap[0][0], rmap[0][1], cv::INTER_LINEAR);
        remap(img2, right_rectified, rmap[1][0], rmap[1][1], cv::INTER_LINEAR);
    }
    else
    {
      if(DEBUG)
	std::cout << "Rectifying without camera parameters..." << std::endl;
        // using uncalibrated stereo rectification
        std::vector<cv::Point2f> points1, points2;
        for (int i = 0; i < good_matches.size(); i++)
        {
            points1.push_back(keypoints1[good_matches[i].queryIdx].pt);
            points2.push_back(keypoints2[good_matches[i].trainIdx].pt);
        }
        if (points1.size() != points2.size() || points1.size() < 8)
        {
            std::cout << "Error: The number of points is not enough." << std::endl;
            return;
        }
        cv::Mat homography1, homography2;
        double threshold = 5.0;
        if (!stereoRectifyUncalibrated(points1, points2, fundamental_matrix, img1.size(), homography1, homography2, threshold))
        {
            std::cout << "Error: Failed to rectify images." << std::endl;
            return;
        }
        warpPerspective(img1, left_rectified, homography1, img1.size());
        warpPerspective(img2, right_rectified, homography2, img2.size());
    }
    return;
}

void compute_disparity_map(cv::Mat left, cv::Mat right, cv::Mat &disp)
{
    cv::Ptr<cv::ximgproc::DisparityWLSFilter> wls_filter;
    cv::Mat left_gray, right_gray, left_disp, right_disp;
    cvtColor(left, left_gray, cv::COLOR_BGR2GRAY);
    cvtColor(right, right_gray, cv::COLOR_BGR2GRAY);

    if (compare_string(DENSE_MATCHING_METHOD, "bm"))
    {
        int max_disp = 160, wsize = 15;
        cv::Ptr<cv::StereoBM> left_matcher = cv::StereoBM::create(max_disp, wsize);
        cv::Ptr<cv::StereoMatcher> right_matcher = cv::ximgproc::createRightMatcher(left_matcher);

        if (!USE_POST_FILTERING)
        {
            left_matcher->compute(left_gray, right_gray, disp);
            return;
        }
        else
        {
            left_matcher->compute(left_gray, right_gray, left_disp);
            right_matcher->compute(right_gray, left_gray, right_disp);
            wls_filter = cv::ximgproc::createDisparityWLSFilter(left_matcher);
        }
    }
    else if (compare_string(DENSE_MATCHING_METHOD, "sgbm"))
    {
        int min_disp = 0, max_disp = 160, wsize = 9;
        cv::Ptr<cv::StereoSGBM> left_matcher = cv::StereoSGBM::create(min_disp, max_disp, wsize);
        left_matcher->setP1(24 * wsize * wsize);
        left_matcher->setP2(96 * wsize * wsize);
        left_matcher->setMode(cv::StereoSGBM::MODE_SGBM);
        cv::Ptr<cv::StereoMatcher> right_matcher = cv::ximgproc::createRightMatcher(left_matcher);

        if (!USE_POST_FILTERING)
        {
            left_matcher->compute(left_gray, right_gray, disp);
            return;
        }
        else
        {
            left_matcher->compute(left_gray, right_gray, left_disp);
            right_matcher->compute(right_gray, left_gray, right_disp);
            wls_filter = cv::ximgproc::createDisparityWLSFilter(left_matcher);
        }
    }
    else
    {
        std::cout << "Error: No such dense matching method." << std::endl;
        return;
    }
    double lambda = 8000.0, sigma = 1.5;
    wls_filter->setLambda(lambda);
    wls_filter->setSigmaColor(sigma);
    wls_filter->filter(left_disp, left, disp, right_disp);
    // disp.convertTo(disp, CV_16U);
    // disp.convertTo(disp, CV_64F, 1.0 / 16.0);
    if (DEBUG >= 2)
        std::cout << "Disparity map: " << disp << std::endl;
    return;
}

void compute_disparity_map_modified(cv::Mat left, cv::Mat right, cv::Mat &disp, cv::Mat &disp_vis, cv::Rect roi1=cv::Rect(), cv::Rect roi2=cv::Rect())
{
    cv::Mat left_gray, right_gray;
    cvtColor(left, left_gray, cv::COLOR_BGR2GRAY);
    cvtColor(right, right_gray, cv::COLOR_BGR2GRAY);
    int min_disp = 5, max_disp = 80, wsize = 9;
    if (compare_string(DENSE_MATCHING_METHOD, "bm"))
    {
        cv::Ptr<cv::StereoBM> left_matcher = cv::StereoBM::create(max_disp, wsize);
        left_matcher->setROI1(roi1);
        left_matcher->setROI2(roi2);
        left_matcher->setPreFilterCap(31);
        left_matcher->setTextureThreshold(10);
        left_matcher->setUniquenessRatio(15);
        left_matcher->setSpeckleWindowSize(100);
        left_matcher->setSpeckleRange(32);
        left_matcher->setDisp12MaxDiff(1);
        if (!USE_POST_FILTERING)
        {
            left_matcher->compute(left_gray, right_gray, disp);
        }
        else
        {
            double lambda = 8000.0, sigma = 1.2;
            cv::Ptr<cv::ximgproc::DisparityWLSFilter> wls_filter = cv::ximgproc::createDisparityWLSFilter(left_matcher);
            wls_filter->setLambda(lambda);
            wls_filter->setSigmaColor(sigma);
            cv::Ptr<cv::StereoMatcher> right_matcher = cv::ximgproc::createRightMatcher(left_matcher);
            cv::Mat left_disp, right_disp;
            left_matcher->compute(left_gray, right_gray, left_disp);
            right_matcher->compute(right_gray, left_gray, right_disp);
            wls_filter->filter(left_disp, left_gray, disp, right_disp);
        }
        cv::ximgproc::getDisparityVis(disp, disp_vis);
        //cv::applyColorMap(disp_vis, disp_vis, cv::COLORMAP_JET);
    }
    else if (compare_string(DENSE_MATCHING_METHOD, "sgbm"))
    {
        cv::Ptr<cv::StereoSGBM> left_matcher = cv::StereoSGBM::create(min_disp, max_disp, wsize);
        left_matcher->setP1(24 * wsize * wsize);
        left_matcher->setP2(96 * wsize * wsize);
        left_matcher->setDisp12MaxDiff(1);
        left_matcher->setUniquenessRatio(15);
        left_matcher->setSpeckleWindowSize(0);
        left_matcher->setSpeckleRange(2);
        left_matcher->setPreFilterCap(63);
        left_matcher->setMode(cv::StereoSGBM::MODE_SGBM);
        if (!USE_POST_FILTERING)
        {
            left_matcher->compute(left, right, disp);
        }
        else
        {
            double lambda = 8000.0, sigma = 1.5;
            cv::Ptr<cv::ximgproc::DisparityWLSFilter> wls_filter = cv::ximgproc::createDisparityWLSFilter(left_matcher);
            wls_filter->setLambda(lambda);
            wls_filter->setSigmaColor(sigma);
            cv::Ptr<cv::StereoMatcher> right_matcher = cv::ximgproc::createRightMatcher(left_matcher);
            cv::Mat left_disp, right_disp;
            left_matcher->compute(left, right, left_disp);
            right_matcher->compute(right, left, right_disp);
            wls_filter->filter(left_disp, left_gray, disp, right_disp);
            disp.setTo(-16, disp == -32768);
        }
        cv::ximgproc::getDisparityVis(disp, disp_vis);
        //cv::applyColorMap(disp_vis, disp_vis, cv::COLORMAP_JET);
    }
    else if (compare_string(DENSE_MATCHING_METHOD, "qd"))
    {
      // cv::Ptr<cv::stereo::QuasiDenseStereo> left_matcher = cv::stereo::QuasiDenseStereo::create(left.size());
      // left_matcher->process(left, right);
      // disp = left_matcher->getDisparity();
      // disp.copyTo(disp_vis);
      return;
    }
    else
    {
        std::cout << "No such dense matching method." << std::endl;
        return;
    }
    double min_z, max_z;
    cv::minMaxLoc(disp, &min_z, &max_z);
    std::cout << "minimum disparity: " << min_z << std::endl << "maximum disparity: " << max_z << std::endl;
    return;
}

void reproject_to_3d(cv::Mat disparityMap, cv::Mat rgb_map, struct cameraParams camParams){
  disparityMap.convertTo(disparityMap, CV_32FC1, 1.0 / 16.0);
  cv::Mat xyz;
  std::cout << "Q_matrix: " << Q_matrix << std::endl;
  Vertex *vertices = new Vertex[disparityMap.rows * disparityMap.cols];
  cv::reprojectImageTo3D(disparityMap, xyz, Q_matrix, true);
  // read x, y, z values
  for (int i = 0; i < xyz.rows; i++)
  {
    for (int j = 0; j < xyz.cols; j++)
    {
      cv::Vec3f point = xyz.at<cv::Vec3f>(i, j);
      double depth = point[2];
      int idx = i * disparityMap.cols + j;
      if (depth != MINF && depth != 0 && depth < 5000 && depth != -1)
	{ // range filter: (0, 1 meter)
	  float X_c = (float(j) - camParams.cX) * depth / camParams.fX;
	  float Y_c = (float(i) - camParams.cY) * depth / camParams.fY;
	  Vector4f P_c = Vector4f(X_c, Y_c, depth, 1);
	  
	  vertices[idx].position = P_c;
	  unsigned char R = rgb_map.at<cv::Vec3b>(i, j)[2];
	  unsigned char G = rgb_map.at<cv::Vec3b>(i, j)[1];
	  unsigned char B = rgb_map.at<cv::Vec3b>(i, j)[0];
	  unsigned char A = 255;
	  vertices[idx].color = Vector4uc(R, G, B, A);
	}
      else
	{
	  vertices[idx].position = Vector4f(MINF, MINF, MINF, MINF);
	  vertices[idx].color = Vector4uc(0, 0, 0, 0);
	}
    }
  }
  // TODO: project the vertices from camera space to world space
  std::stringstream ss;
  ss << PROJECT_PATH << "Output/pointclouds/reproject-3d.off";
  writeMesh(vertices, disparityMap.cols, disparityMap.rows, ss.str());
  return;
}

void get_depth_map_from_disparity_map(cv::Mat disparityMap, struct cameraParams camParams, cv::Mat &depthMap)
{
  // TODO: the disparity map are calculated on rectified images, so the depth map should be calculated on rectified images
  // if use original images, the depth map will be wrong, triangulation should be used to calculate the depth map
    if (USE_GROUNDTRUTH == 1)
    {
        std::string groundtruth_disparity_map_dir = PROJECT_PATH + "Data/kitti" + "/data_scene_flow/training/disp_noc_0/";
        std::string image_name = "000015_10.png";
        cv::Mat disparity_Map = cv::imread(groundtruth_disparity_map_dir + image_name, cv::IMREAD_UNCHANGED);
        int width = disparity_Map.cols, height = disparity_Map.rows;
        depthMap = cv::Mat(height, width, CV_64F);
        for (int h = 0; h < height; h++)
        {
            for (int w = 0; w < width; w++)
            {
                uint16_t disparity = disparity_Map.at<uint16_t>(h, w);
                if (disparity == 0)
                    depthMap.at<double>(h, w) = 0;
                else{
                    depthMap.at<double>(h, w) = (256.0 * camParams.fX * camParams.baseline) / (float)disparity;
		}
            }
        }
        return;
    }
    int width = disparityMap.cols;
    int height = disparityMap.rows;
    depthMap = cv::Mat(height, width, CV_64F);
    for (int h = 0; h < height; h++)
    {
        for (int w = 0; w < width; w++)
        {
            uint16_t disparity = disparityMap.at<uint16_t>(h, w);
            if (disparity == 0)
                depthMap.at<double>(h, w) = 0;
            else{
	      // depthMap.at<double>(h, w) = 256.0 * camParams.fX * camParams.baseline / (disparity * 6000);
	      depthMap.at<double>(h, w) = (16.0 * camParams.fX * camParams.baseline) / (float)disparity;
	    }
        }
    }
    return;
}

void get_point_cloud_from_depth_map(cv::Mat depth_map, cv::Mat rgb_map, struct cameraParams camParams, std::string filename)
{
  if (USE_GROUNDTRUTH == 2){
    std::string groundtruth_depth_map = PROJECT_PATH + "Data/bricks-rgbd" + "/frame-000755.depth.png";
    cv::Mat depth_image = cv::imread(groundtruth_depth_map, cv::IMREAD_UNCHANGED);
    depth_image.convertTo(depth_image, CV_32FC1, 1.0 / 1000.0);
    
    std::string color_image_path = PROJECT_PATH + "Data/" + DATASET + "/frame-000755.color.png";
    cv::Mat color_image = cv::imread(color_image_path, cv::IMREAD_UNCHANGED);

    cv::Mat map_x, map_y;
    cv::Mat depth_intrinsics = (cv::Mat_<float>(3, 3) <<
        577.871, 0, 319.623,
        0, 580.258, 239.624,
        0, 0, 1);
    // Get the intrinsic parameters of the color camera
    cv::Mat color_intrinsics = (cv::Mat_<float>(3, 3) <<
        1170.19, 0, 647.75,
        0, 1170.19, 483.75,
        0, 0, 1);
    depth_intrinsics.convertTo(depth_intrinsics, CV_32F);
    color_intrinsics.convertTo(color_intrinsics, CV_32F);
    // cv::resize(depth_image, color_image, cv::Size(color_image.cols, color_image.rows));
    cv::initUndistortRectifyMap(depth_intrinsics, cv::Mat::zeros(5, 1, CV_32F), cv::Mat::eye(3, 3, CV_32F), color_intrinsics, color_image.size(), CV_16SC2, map_x, map_y);
    // cv::initUndistortRectifyMap(color_intrinsics, cv::Mat::zeros(5, 1, CV_32F), cv::Mat::eye(3, 3, CV_32F), depth_intrinsics, depth_image.size(), CV_16SC2, map_x, map_y);
    // Align the depth image to the color image
    cv::Mat depth_aligned;
    // cv::Mat color_aligned;
    cv::remap(depth_image, depth_aligned, map_x, map_y, cv::INTER_LINEAR);
    // cv::remap(color_image, color_aligned, map_x, map_y, cv::INTER_LINEAR);
    depth_aligned.convertTo(depth_aligned, CV_64F);
    // color_aligned.convertTo(color_aligned, CV_64F);
    depth_map = depth_aligned;
    // depth_map = depth_image;
    // rgb_map = color_aligned;
    rgb_map = color_image;
    std::cout << "rgb map size: " << rgb_map.size() << std::endl;
    std::cout << "depth map size: " << depth_map.size() << std::endl;

    camParams.cX = 647.75;
    camParams.cY = 483.75;
    camParams.fX = 1170.19;
    camParams.fY = 1170.19;
  }

    int width = depth_map.cols;
    int height = depth_map.rows;

    Vertex *vertices = new Vertex[width * height];
    double max_depth = 0.0;
    double min_depth = 100000.0;
    double average_depth = 0.0;
    int valid_depth_count = 0;
    for (int h = 0; h < height; h++)
    {
        for (int w = 0; w < width; w++)
        {
            int idx = h * width + w;
            float depth = (float)(depth_map.at<double>(h, w));
	    // TODO: notice the storage format of cv::Mat, float number must read with float, integer must
	    // be read by integer type, otherwise the value would be 0.
	    // float depth = (float)(depth_map.at<uint16_t>(h, w));
            if (depth != MINF && depth != 0 && depth < 5000 && depth != -1)
            { // range filter: (0, 1 meter)
	      double X_c = (double(w) - camParams.cX) * depth / camParams.fX;
	      double Y_c = (double(h) - camParams.cY) * depth / camParams.fY;

	      cv::Mat point = (cv::Mat_<double>(4, 1) << X_c, Y_c, depth, 1);
	      point.convertTo(point, CV_64FC1);
	      // std::cout << "point: " << point << std::endl;
	      // std::cout << "rgb map: " << camParams.left_camera_extrinsic_reverse << std::endl;
	      cv::Mat point_world = camParams.left_camera_extrinsic_reverse * point;
	      X_c = point_world.at<double>(0, 0);
	      Y_c = point_world.at<double>(1, 0);
	      depth = point_world.at<double>(2, 0);
	      // TODO: DON"T CONVERT DOUBLE TO FLOAT IMPLICITLY, IT WILL CAUSE ERROR
	      // std::cout << "point world: " << point_world << std::endl;
	      // std::cout << "x y z: " << X_c << " " << Y_c << " " << depth << std::endl;
	      
	      if(depth > max_depth)
		max_depth = depth;
	      if(min_depth > depth)
		min_depth = depth;
	      average_depth += depth;
	      valid_depth_count++;
	      
	      Vector4f P_c = Vector4f(X_c, Y_c, depth, 1);

	      vertices[idx].position = P_c;
	      unsigned char R = rgb_map.at<cv::Vec3b>(h, w)[2];
	      unsigned char G = rgb_map.at<cv::Vec3b>(h, w)[1];
	      unsigned char B = rgb_map.at<cv::Vec3b>(h, w)[0];
	      unsigned char A = 255;
	      vertices[idx].color = Vector4uc(R, G, B, A);
            }
            else
            {
                vertices[idx].position = Vector4f(MINF, MINF, MINF, MINF);
                vertices[idx].color = Vector4uc(0, 0, 0, 0);
            }
        }
    }
    std::cout << "max depth: " << max_depth << std::endl;
    std::cout << "min depth: " << min_depth << std::endl;
    std::cout << "average depth: " << average_depth / valid_depth_count << std::endl;

    // write to the off file
    std::stringstream ss;
    ss << PROJECT_PATH << "Output/pointclouds/" << filename << ".off";
    writeMesh(vertices, width, height, ss.str());
    return;
}

bool icp_reconstruct(const std::string base_model, const std::string other_model, std::string &target_model)
{
    SimpleMesh sourceMesh;
    if (!sourceMesh.loadMesh(base_model))
    {
        std::cout << "Mesh file wasn't read successfully at location: " << base_model << std::endl;
        return false;
    }

    SimpleMesh targetMesh;
    if (!targetMesh.loadMesh(other_model))
    {
        std::cout << "Mesh file wasn't read successfully at location: " << other_model << std::endl;
        return false;
    }

    // Estimate the pose from source to target mesh with ICP optimization.
    ICPOptimizer *optimizer = nullptr;
    if (USE_LINEAR_ICP)
    {
        std::cout << "USING LINEAR ICP" << std::endl;
        optimizer = new LinearICPOptimizer();
    }
    else
    {
        optimizer = new CeresICPOptimizer();
    }

    optimizer->setMatchingMaxDistance(0.0003f);
    if (USE_POINT_TO_PLANE)
    {
        std::cout << "USING POINT TO PLANE" << std::endl;
        optimizer->usePointToPlaneConstraints(true);
        optimizer->setNbOfIterations(10);
    }
    else
    {
        optimizer->usePointToPlaneConstraints(false);
        optimizer->setNbOfIterations(20);
    }

    PointCloud source{sourceMesh};
    PointCloud target{targetMesh};

    Matrix4f estimatedPose = Matrix4f::Identity();
    optimizer->estimatePose(source, target, estimatedPose);

    // Visualize the resulting joined mesh. We add triangulated spheres for point matches.
    SimpleMesh resultingMesh = SimpleMesh::joinMeshes(sourceMesh, targetMesh, estimatedPose);
    resultingMesh.writeMesh(target_model);

    delete optimizer;

    return true;
}

void merge(std::string models_directory)
{
    // this is using frame to frame reconstruction method
    DIR *directory = opendir(models_directory.c_str());
    struct dirent *entry;
    int index = 0;
    std::string base_model, target_model, other_model;
    if (compare_string(MERGE_METHOD, "frame-to-frame"))
    {
        while (directory != NULL)
        {
            while ((entry = readdir(directory)) != NULL)
            {
                if (index == 0)
                {
                    base_model = models_directory + std::string(entry->d_name);
                    index++;
                    continue;
                }
                other_model = models_directory + std::string(entry->d_name);
                target_model = models_directory + std::to_string(index) + ".off";
                if (!icp_reconstruct(base_model, other_model, target_model))
                {
                    std::cout << "Error: Failed to reconstruct model. Skipped one model: " + base_model << std::endl;
                    base_model = other_model;
                }
                else
                    base_model = target_model;
                index++;
            }
        }
    }
    else if (compare_string(MERGE_METHOD, "frame-to-model"))
    {
        // this is using frame-to-model method
        std::vector<std::string> models;
        while (directory != NULL)
        {
            while ((entry = readdir(directory)) != NULL)
            {
                models.push_back(models_directory + std::string(entry->d_name));
            }
        }
        std::string base_model = models[0];
        for (int i = 1; i < models.size(); i++)
        {
            target_model = models_directory + std::to_string(i) + ".off";
            if (!icp_reconstruct(base_model, models[i], target_model))
            {
                std::cout << "Error: Failed to reconstruct model. Skipped one model: " + models[i] << std::endl;
            }
            else
                base_model = target_model;
        }
    }
    closedir(directory);
}

std::string flag_to_string(int flag){
  if(flag)
    return "True";
  else
    return "False";
}

void print_running_information(){
  std::cout << "Using dataset:                         " << DATASET << std::endl;
  std::cout << "Test mode:                             " << flag_to_string(TEST) << std::endl;
  if(USE_GROUNDTRUTH > 0){
    if(USE_GROUNDTRUTH == 1)
      std::cout << "Using groundtruth disparity map of kitti..." << std::endl;
    else
      std::cout << "Using groudtruth depth map of bricks-rgbd..." << std::endl;
  }
  std::cout << "Debug mode                             " << flag_to_string(DEBUG) << std::endl;
  std::cout << "Descriptor method:                     " << DESCRIPTOR_METHOD << std::endl;
  std::cout << "Descriptor matching method:            " << DESCRIPTOR_MATCHING_METHOD << std::endl;
  std::cout << "Fundamental matrix calculation method: " << FUNDAMENTAL_MATRIX_METHOD << std::endl;
  std::cout << "Dense matching method:                 " << DENSE_MATCHING_METHOD << std::endl;
  std::cout << "Use post filtering:                    " << flag_to_string(USE_POST_FILTERING) << std::endl;
  if(RECONSTRUCT){
    std::cout << "Reconstruct the whole 3D model:        " << flag_to_string(RECONSTRUCT) << std::endl;
    std::cout << "Use linear icp:                        " << flag_to_string(USE_LINEAR_ICP) << std::endl;
    std::cout << "Use point to plane distance in icp:    " << flag_to_string(USE_POINT_TO_PLANE) << std::endl;
    std::cout << "Merging model method:                  " << MERGE_METHOD << std::endl;
  }
  std::cout << "Compute dense matching performance:    " << flag_to_string(COMPARE_DENSE_MATCHING) << std::endl; 
}
