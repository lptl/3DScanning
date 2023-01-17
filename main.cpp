#include <dirent.h>

#include "Libs/Pipeline.h"

#define DATASET "kitti" // kitti, bricks-rgbd
#define TEST 1 // 0: no test, 1: test
#define RECONSTRUCT 0 // 0: no reconstruction, 1: reconstruction

void process_pair_images(std::string filename1, std::string filename2, struct cameraParams camParams)
{
    cv::Mat left = imread(filename1, cv::IMREAD_COLOR);
    cv::Mat right = imread(filename2, cv::IMREAD_COLOR);
    // TODO: calibrate image distortion if needed
    if(left.empty() || right.empty()){
        std::cout << "Error: Image not found or failed to open image." << std::endl;
        return;
    }

    std::string img1_name = filename1.substr(filename1.find_last_of("/\\") + 1);
    std::string img2_name = filename2.substr(filename2.find_last_of("/\\") + 1);

    std::cout << "Processing " << filename1 << " and " << filename2 << std::endl;

    std::cout << "Detecting " << DESCRIPTOR_METHOD << " keypoints/features for images" << std::endl;
    struct detectResult result1, result2;
    detect_keypoints_or_features(img1_name, left, &result1);
    detect_keypoints_or_features(img2_name, right, &result2);
    /*
    Mat keypoints_on_image;
    drawKeypoints(left, result1.keypoints, keypoints_on_image, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    imshow("Keypoints", keypoints_on_image);
    waitKey();
    */
    
    std::cout << "Matching descriptors for images using " << DESCRIPTOR_MATCHING_METHOD << std::endl;
    std::vector<cv::DMatch> good_matches;
    match_descriptors(result1.descriptors, result2.descriptors, good_matches);
    // save to file
    cv::Mat img_matches;
    drawMatches(left, result1.keypoints, right, result2.keypoints, good_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    imwrite(PROJECT_PATH + "Output/descriptor_match/" + std::to_string(result1.filetype.number) + "-" + std::to_string(result2.filetype.number) + ".png", img_matches);

    std::cout << "Finding fundamental matrix for images using " << FUNDAMENTAL_MATRIX_METHOD << std::endl;
    cv::Mat fundamental_matrix;
    find_fundamental_matrix(result1.keypoints, result2.keypoints, good_matches, fundamental_matrix);
    //std::cout << fundamental_matrix << std::endl;

    std::cout << "Rectifying images" << std::endl;
    cv::Mat left_rectified, right_rectified;
    rectify_images(left, right, result1.keypoints, result2.keypoints, good_matches, fundamental_matrix, camParams, left_rectified, right_rectified);
    // save to file
    imwrite(PROJECT_PATH + "Output/left_rectified/" + img1_name, left_rectified);
    imwrite(PROJECT_PATH + "Output/right_rectified/" + img2_name, right_rectified);
    
    std::cout << "Computing disparity map for images using " << DENSE_MATCHING_METHOD << std::endl;
    cv::Mat disp;
    compute_disparity_map(left_rectified, right_rectified, disp);
    // save to file
    imwrite(PROJECT_PATH + "Output/disparity_map.png" , disp);

    std::cout << "Generating point cloud" << std::endl;
    cv::Mat depthMap = cv::Mat::zeros(disp.rows, disp.cols, CV_16UC1);
    get_depth_map_from_disparity_map(disp, camParams, depthMap);
    get_point_cloud_from_depth_map(depthMap, left_rectified, camParams, std::to_string(result1.filetype.number));


    std::cout << "Finished processing " << filename1 << " and " << filename2 << std::endl << std::endl;
    return;
}

int main()
{
    std::string dataset_dir = PROJECT_PATH + "Data/" + DATASET;
    if (compare_string(DATASET, "bricks-rgbd")) {
        DIR* directory = opendir(dataset_dir.c_str());
        struct dirent* entry;
        if (directory == NULL) {
            std::cout << "Error in main.cpp main(): Directory not found or failed to open directory." << dataset_dir << std::endl;
            return -1;
        }
        else {
            struct cameraParams camParams;
            while ((entry = readdir(directory)) != NULL) {
                if (entry->d_name[0] != 'f')
                    continue;
                struct filenameType filename_type = extract_file_name(entry->d_name);
                if (filename_type.category != 0)
                    continue;
                process_pair_images(dataset_dir + "/" + entry->d_name, dataset_dir + "/" + get_file_name(filename_type.number + 1, filename_type.category), camParams);
                if(TEST)
                    break;
            }
        }
        if(RECONSTRUCT)
            reconstruct(MODELS_DIR);
        closedir(directory);
    }
    else if (compare_string(DATASET, "kitti")) {
        std::string left_dir = dataset_dir + "/data_scene_flow/training/image_2/";
        std::string right_dir = dataset_dir + "/data_scene_flow/training/image_3/";
        std::string calib_dir = dataset_dir + "/data_scene_flow_calib/training/calib_cam_to_cam/";
        DIR* directory = opendir(left_dir.c_str());
        struct dirent* entry;
        if (directory == NULL) {
            std::cout << "Error in main.cpp main(): Directory not found or failed to open directory." << left_dir << std::endl;
            return -1;
        }
        else {
            while ((entry = readdir(directory)) != NULL) {
                if (entry->d_name[0] == '.')
                    continue;
                std::string filename = entry->d_name;
                std::string calib_file = calib_dir + filename.substr(filename.find_last_of("/") + 1, filename.find("_")) + ".txt";
                struct cameraParams camParams;
                getCameraParamsKITTI(calib_file, &camParams);
                //std::cout << camParams.left_camera_matrix << std::endl;
                //std::cout << camParams.left_distortion_coeffs << std::endl;
                //std::cout << camParams.right_camera_matrix << std::endl;
                //std::cout << camParams.right_distortion_coeffs << std::endl;
                //std::cout << camParams.left_to_right_R << std::endl;
                //std::cout << camParams.left_to_right_T << std::endl;
                process_pair_images(left_dir + filename, right_dir + filename, camParams);
                if(TEST)
                    break;
            }
        }
        closedir(directory);
        if(RECONSTRUCT)
            reconstruct(MODELS_DIR);
    }

    std::cout << "Stereo Reconstruction Finished" << std::endl;
    return 0;
}