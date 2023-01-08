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

#include <iostream>
#include <fstream>
#include <array>
#include <iomanip>

using namespace cv;
using namespace std;

struct detectResult {
    vector<KeyPoint> keypoints;
    Mat descriptors;
    string framenumber;
};

struct detectResult detectDescriptors(string, Mat);
void processImagePair(string, string, string);
vector<DMatch> matchDescriptors(Mat, Mat, struct detectResult, struct detectResult);
Mat findFundamentalMatrix(vector<KeyPoint>, vector<KeyPoint>, vector<DMatch>);
void rectifyImages(Mat, Mat, vector<KeyPoint>, vector<KeyPoint>, vector<DMatch>, Mat);
string padZeroes(const int, const int);

#define DESCRIPTOR_METHOD "fast" // shi-tomasi, fast, harris, sift, surf, orb, brisk, kaze, akaze
#define DESCRIPTOR_MATCHING_METHOD "flann" // flann, brute_force
#define FUNDAMENTAL_MATRIX_METHOD "ransac" // ransac, lmeds, 7point, 8point


void processImagePair(string datasetDir, string filename1, string filename2) {
    
    Mat img1 = imread(datasetDir + filename1, IMREAD_COLOR);
    Mat img2 = imread(datasetDir + filename2, IMREAD_COLOR);
    
    // TODO: calibrate image distortion if needed
    if (img1.empty() || img2.empty()) {
        cout << "Error: Image not found or failed to open image." << endl;
        return;
    }

    cout << "detecting keypoints and features for image " << filename1 << " and " << filename2 << endl;
    struct detectResult result1 = detectDescriptors(filename1, img1);
    struct detectResult result2 = detectDescriptors(filename2, img2);

    cout << "Processing " << filename1 << " and " << filename2 << endl;
    cout << "Matching descriptors for image " << endl;
    vector<DMatch> correspondences = matchDescriptors(img1, img2, result1, result2);

    cout << "Finding fundamental matrix for image " << endl;
    Mat fundamental_matrix = findFundamentalMatrix(result1.keypoints, result2.keypoints, correspondences);
    // cout << fundamental_matrix << endl;

    cout << "Rectifying images for image " << endl;
    rectifyImages(img1, img2, result1.keypoints, result2.keypoints, correspondences, fundamental_matrix);
    
    // use rectified images to do stereo matching
<<<<<<< Updated upstream
    // https://docs.opencv.org/4.6.0/d2/d6e/classcv_1_1StereoMatcher.html#a03f7087df1b2c618462eb98898841345
    cout << "Finished processing " << filename1 << " and " << filename2 << endl;
    return;
}

void rectifyImages(Mat img1, Mat img2, vector<KeyPoint> keypoints1, vector<KeyPoint> keypoints2, vector<DMatch> correspondences, Mat fundamental_matrix) {
    
    vector<Point2f> points1, points2;
    for (int i = 0; i < correspondences.size(); i++) {
        points1.push_back(keypoints1[correspondences[i].queryIdx].pt);
        points2.push_back(keypoints2[correspondences[i].trainIdx].pt);
    }
    if (points1.size() != points2.size() || points1.size() < 8) {
        cout << "Error: The number of points is not enough." << endl;
        cout << points1.size() << endl;
        cout << points2.size() << endl;
        return;
    }

    Mat homography1, homography2;
    double threshold = 5.0;
    // TODO: the following function doesn't require intrinsic paramters, but we have intrinsic parameters to be used
    if (!stereoRectifyUncalibrated(points1, points2, fundamental_matrix, img1.size(), homography1, homography2, threshold)) {
        cout << "Error: Failed to rectify images." << endl;
        return;
    }
    Mat rectified1, rectified2;
    // TODO: rectify images
    warpPerspective(img1, rectified1, homography1, img1.size());
    warpPerspective(img2, rectified2, homography2, img2.size());

=======
    // https://docs.opencv.org/3.4/d2/d6e/classcv_1_1StereoMatcher.html#a03f7087df1b2c618462eb98898841345
    std::cout << "Computing disparity map for image " << std::endl;
    compute_disparity_map(img1, img2);
    std::cout << "Finished processing " << filename1 << " and " << filename2 << std::endl;
>>>>>>> Stashed changes
    return;
}

Mat findFundamentalMatrix(vector<KeyPoint> keypoints1, vector<KeyPoint> keypoints2, vector<DMatch> correspondences) {
    
    vector<Point2f> points1, points2;
    for (int i = 0; i < correspondences.size(); i++) {
        points1.push_back(keypoints1[correspondences[i].queryIdx].pt);
        points2.push_back(keypoints2[correspondences[i].trainIdx].pt);
    }
    if (points1.size() != points2.size() || points1.size() < 8) {
        cout << "Error: The number of points is not enough." << endl;
        return Mat();
    }
    
    Mat fundamental_matrix;
    double ransacReprojThreshold = 3.0, confidence = 0.99;
    if (FUNDAMENTAL_MATRIX_METHOD == "ransac")
        fundamental_matrix = findFundamentalMat(points1, points2, FM_RANSAC, ransacReprojThreshold, confidence);
    else if (FUNDAMENTAL_MATRIX_METHOD == "lmeds")
        fundamental_matrix = findFundamentalMat(points1, points2, FM_LMEDS, ransacReprojThreshold, confidence);
    else if (FUNDAMENTAL_MATRIX_METHOD == "7point")
        fundamental_matrix = findFundamentalMat(points1, points2, FM_7POINT);
    else if (FUNDAMENTAL_MATRIX_METHOD == "8point")
        fundamental_matrix = findFundamentalMat(points1, points2, FM_8POINT);
    else {
        cout << "Error: No such fundamental matrix method." << endl;
        return Mat();
    }
    return fundamental_matrix;
}

vector<DMatch> matchDescriptors(Mat img1, Mat img2, struct detectResult result1, struct detectResult result2) {
    
    Mat descriptors1 = result1.descriptors, descriptors2 = result2.descriptors;
    vector<KeyPoint> keypoints1 = result1.keypoints, keypoints2 = result2.keypoints;
    vector<vector<DMatch>> correspondences;
    
    if (DESCRIPTOR_MATCHING_METHOD == "flann") {
        Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
        // the knn use NORM_L2 because sift is a float-pointing descriptor
        // descriptor1 = queryDescriptor, descriptor2 = trainDescriptor
        matcher->knnMatch(descriptors1, descriptors2, correspondences, 2);
    }
    else if (DESCRIPTOR_MATCHING_METHOD == "brute_force") {
        Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);
        // the knn use NORM_L2 because sift is a float-pointing descriptor
        matcher->knnMatch(descriptors1, descriptors2, correspondences, 2);
    }
    else {
        cout << "Error: No such matching method." << endl;
        exit(-1);
    }

    const float ratio_thresh = 0.7f;
    vector<DMatch> good_matches;
    for (size_t i = 0; i < correspondences.size(); i++) {
        if (correspondences[i][0].distance < ratio_thresh * correspondences[i][1].distance)
            good_matches.push_back(correspondences[i][0]);
    }

    Mat img_matches;
    drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    imwrite("./matches/" + result1.framenumber + "-" + result2.framenumber + ".png", img_matches);

    return good_matches;
}

struct detectResult detectDescriptors(string filename, Mat img) {
    
    if (DESCRIPTOR_METHOD == "shi-tomasi") {
        vector<Point2f> corners;

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

        vector<KeyPoint> keypoints;
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
        result.framenumber = filename.substr(10, 3);
        return result;
    }
    else if (DESCRIPTOR_METHOD == "fast") {
        Ptr<FastFeatureDetector> fastDetector = FastFeatureDetector::create();
        
        Mat img_gray;
        cvtColor(img, img_gray, COLOR_BGR2GRAY);
        vector<KeyPoint> keypoints;

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
        result.framenumber = filename.substr(10, 3);
        return result;
    }
}

string padZeroes(const int i, const int length) {
    
    ostringstream ostr;
    if (i < 0)
        ostr << '-';

    ostr << setfill('0') << setw(length) << (i < 0 ? -i : i);
    return ostr.str();
}

int main()
{
<<<<<<< Updated upstream
    string datasetDir = "E:/Study/Courses/3DScanningMotionCapture/Project/Data/bricks-rgbd/";

    int currentIdx = 0, increment = 1, maxIdx = 772;

    while (currentIdx < maxIdx) {
        string filename1 = "frame-000" + padZeroes(currentIdx, 3) + ".color.png";
        string filename2 = "frame-000" + padZeroes(currentIdx + 1, 3) + ".color.png";
        
        processImagePair(datasetDir, filename1, filename2);
        
        currentIdx = currentIdx + increment;
=======
    std::string dataset_dir = PROJECT_PATH + "Data/bricks-rgbd/";
    DIR* directory = opendir(dataset_dir.c_str());
    struct dirent* entry;
    if (directory == NULL) {
        std::cout << "Error in main.cpp main(): Directory not found or failed to open directory." << std::endl;
        return -1;
    }
    else {
        while ((entry = readdir(directory)) != NULL) {
            if (entry->d_name[0] != 'f')
                continue;
            struct filenameType filename_type = extract_file_name(entry->d_name);
            if (filename_type.category != 0)
                continue;
            process_pair_images(dataset_dir, entry->d_name, get_file_name(filename_type.number + 1, filename_type.category));
        }
        closedir(directory);
>>>>>>> Stashed changes
    }

    cout << "Stereo Reconstruction Finished" << endl;
    return 0;
}