#include <dirent.h>

#include "Libs/Pipeline.h"
#include "Libs/Evaluate.h"

#define DATASET "kitti"             // kitti, bricks-rgbd
#define TEST 0                      // 0: no test, 1: test
#define RECTIFY 0                   // 0: use groundtruth, 1: rectify based on sparse matching
#define RECONSTRUCT 0               // 0: no reconstruction, 1: reconstruction

void process_pair_images(std::string filename1, std::string filename2, struct cameraParams camParams)
{
    //std::cout << filename1 << std::endl << filename2 << std::endl;
    cv::Mat left = imread(filename1, cv::IMREAD_COLOR);
    cv::Mat right = imread(filename2, cv::IMREAD_COLOR);

    if (left.empty() || right.empty())
    {
        std::cout << "Error: Image not found or failed to open image." << std::endl;
        return;
    }

    std::string img1_name = filename1.substr(filename1.find_last_of("/") + 1);
    std::string img2_name = filename2.substr(filename2.find_last_of("/") + 1);

    cv::Mat left_rectified, right_rectified;
    cv::Rect roi1 = cv::Rect(), roi2 = cv::Rect();

    std::cout << "Processing " << filename1 << " and " << filename2 << std::endl;

    if (RECTIFY)
    {
        cv::Mat left_undistorted, right_undistorted;
        cv::undistort(left, left_undistorted, camParams.left_camera_matrix, camParams.left_distortion_coeffs);
        cv::undistort(right, right_undistorted, camParams.right_camera_matrix, camParams.right_distortion_coeffs);

        std::cout << "Detecting keypoints and computing descriptors" << std::endl;
        struct detectResult result1, result2;
        detect_keypoints_or_features(img1_name, left_undistorted, &result1);
        detect_keypoints_or_features(img2_name, right_undistorted, &result2);

        std::cout << "Matching descriptors" << std::endl;
        std::vector<cv::DMatch> good_matches;
        match_descriptors(result1.descriptors, result2.descriptors, good_matches);
        cv::Mat img_matches;
        cv::drawMatches(left_undistorted, result1.keypoints, right_undistorted, result2.keypoints, good_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        cv::imwrite(PROJECT_PATH + "Output/descriptor_match/" + std::to_string(result1.filetype.number) + "-" + std::to_string(result2.filetype.number) + ".png", img_matches);

        std::cout << "Computing fundamental matrix" << std::endl;
        cv::Mat fundamental_matrix, mask;
        std::vector<cv::Point2f> points1, points2;
        find_fundamental_matrix(result1.keypoints, result2.keypoints, good_matches, fundamental_matrix, mask, points1, points2);
        cv::Mat img_lines;
        draw_epipolar_lines(left_undistorted, right_undistorted, fundamental_matrix, points1, points2, mask, img_lines);
        cv::imwrite(PROJECT_PATH + "Output/epipolar_lines/" + std::to_string(result1.filetype.number) + "-" + std::to_string(result2.filetype.number) + ".png", img_lines);

        std::cout << "Rectifying images" << std::endl;
        rectify_images(left, right, points1, points2, fundamental_matrix, mask, camParams, left_rectified, right_rectified, roi1, roi2);
        cv::imwrite(PROJECT_PATH + "Output/left_rectified/" + img1_name, left_rectified);
        cv::imwrite(PROJECT_PATH + "Output/right_rectified/" + img2_name, right_rectified);
    }
    else
    {
        left.copyTo(left_rectified);
        right.copyTo(right_rectified);
    }
    
    std::string img1_num = img1_name.substr(0, img1_name.find_last_of("."));
    // Test only the 10th image (for which we have ground truth dis
    if (std::stoi(img1_num) != 10)
    {
        return;
    }

    std::cout << "Computing disparity map" << std::endl;
    cv::Mat disp, disp_vis;
    compute_disparity_map(left_rectified, right_rectified, disp, disp_vis, roi1, roi2);
    cv::imwrite(PROJECT_PATH + "Output/disparity_map/" + img1_name, disp_vis);
    

    if (compare_string(DATASET, "kitti") && std::stoi(img1_num) == 10)
    {
        std::string dataset_dir = filename1.substr(0, filename1.find("kitti") + 5);
        cv::Mat gt_disp = imread(dataset_dir + "/disparity_map/" + img1_name, cv::IMREAD_UNCHANGED);
        gt_disp.convertTo(gt_disp, CV_32F, 1.0f / 256.0f);
        gt_disp.setTo(-1, gt_disp == 0);

        cv::Mat err_img;
        disp_error(gt_disp, disp, err_img);
        cv::imwrite(PROJECT_PATH + "Output/disparity_map/" + img1_num + "_error.png", err_img);
    }

    std::cout << "Generating point cloud" << std::endl;
    cv::Mat depthMap;
    get_depth_map_from_disparity_map(disp, camParams, depthMap);
    get_point_cloud_from_depth_map(depthMap, left, camParams, PROJECT_PATH + "Output/pointclouds/" + img1_num + ".off");

    std::cout << "Finished processing " << filename1 << " and " << filename2 << std::endl
              << std::endl;
    return;
}

int main()
{
    std::cout << "Using dataset:                         " << DATASET << std::endl;
    std::cout << "Test mode:                             " << TEST << std::endl;
    std::cout << "Debug mode                             " << DEBUG << std::endl;
    std::cout << "Descriptor method:                     " << DESCRIPTOR_METHOD << std::endl;
    std::cout << "Matching method:                       " << DESCRIPTOR_MATCHING_METHOD << std::endl;
    std::cout << "Fundamental matrix calculation method: " << FUNDAMENTAL_MATRIX_METHOD << std::endl;
    std::cout << "Dense matching method:                 " << DENSE_MATCHING_METHOD << std::endl;
    std::cout << "Post filtering after dense matching:   " << USE_POST_FILTERING << std::endl;
    std::cout << "Use linear icp:                        " << USE_LINEAR_ICP << std::endl;
    std::cout << "Use point to plane distance in icp:    " << USE_POINT_TO_PLANE << std::endl;
    std::cout << "Reconstruct the whole 3D model:        " << RECONSTRUCT << std::endl;
    std::cout << "Merging method:                        " << MERGE_METHOD << std::endl;
    
    std::string dataset_dir = PROJECT_PATH + "Data/" + DATASET;
    if (compare_string(DATASET, "bricks-rgbd"))
    {
        DIR *directory = opendir(dataset_dir.c_str());
        struct dirent *entry;
        if (directory == NULL)
        {
            std::cout << "Error in main.cpp main(): Directory not found or failed to open directory." << dataset_dir << std::endl;
            return -1;
        }
        else
        {
            struct cameraParams camParams;
            while ((entry = readdir(directory)) != NULL)
            {
                if (entry->d_name[0] != 'f')
                    continue;
                struct filenameType filename_type = extract_file_name(entry->d_name);
                if (filename_type.category != 0)
                    continue;
                process_pair_images(dataset_dir + "/" + entry->d_name, dataset_dir + "/" + get_file_name(filename_type.number + 1, filename_type.category), camParams);
                if (TEST)
                    break;
            }
        }
        closedir(directory);
    }
    else if (compare_string(DATASET, "kitti"))
    {
        std::string calib_file = dataset_dir + "/calib_cam_to_cam.txt";
        std::string left_dir, right_dir;

        if (RECTIFY)
        {
            left_dir = dataset_dir + "/unrectified/image_02/data/";
            right_dir = dataset_dir + "/unrectified/image_03/data/";
        }
        else
        {
            left_dir = dataset_dir + "/rectified/image_02/data/";
            right_dir = dataset_dir + "/rectified/image_03/data/";
        }
        
        DIR *left_directory = opendir(left_dir.c_str());
        DIR* right_directory = opendir(right_dir.c_str());
        struct dirent *entry_left, *entry_right;
        if (left_directory == NULL || right_directory == NULL)
        {
            std::cout << "Error in main.cpp main(): Directory not found or failed to open directory." << left_dir << std::endl << right_dir << std::endl;
            return -1;
        }
        else
        {
            struct cameraParams camParams;
            getCameraParamsKITTI(calib_file, &camParams);

            while ((entry_left = readdir(left_directory)) != NULL && (entry_right = readdir(right_directory)) != NULL)
            {
                if (entry_left->d_name[0] == '.')
                    continue;
                
                process_pair_images(left_dir + entry_left->d_name, right_dir + entry_right->d_name, camParams);
                if (TEST)
                    break;
            }
        }
        closedir(left_directory);
        closedir(right_directory);
    }
    if (RECONSTRUCT)
        merge(MODELS_DIR);

    std::cout << "Stereo Reconstruction Finished" << std::endl;
    return 0;
}