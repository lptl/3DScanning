#include "Libs/Pipeline.h"
#include "Libs/Evaluate.h"


void process_pair_images(std::string filename1, std::string filename2, struct cameraParams camParams)
{
    cv::Mat left = imread(filename1, cv::IMREAD_COLOR);
    cv::Mat right = imread(filename2, cv::IMREAD_COLOR);
    // TODO: do image distortion calibrration if needed
    if (left.empty() || right.empty())
    {
        std::cout << "Error: Image not found or failed to open image." << std::endl;
        return;
    }

    std::string img1_name = filename1.substr(filename1.find_last_of("/\\") + 1);
    std::string img2_name = filename2.substr(filename2.find_last_of("/\\") + 1);

    std::cout << "Processing " << filename1 << " and " << filename2 << std::endl;
    
    if(DEBUG)
      std::cout << "Detecting keypoints and computing descriptors..." << std::endl;
    struct detectResult result1, result2;
    detect_keypoints_or_features(img1_name, left, &result1);
    detect_keypoints_or_features(img2_name, right, &result2);
    
    if(DEBUG)
      std::cout << "Matching descriptors..." << std::endl;
    std::vector<cv::DMatch> good_matches;
    match_descriptors(result1.descriptors, result2.descriptors, good_matches);
    cv::Mat img_matches;
    drawMatches(left, result1.keypoints, right, result2.keypoints, good_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    imwrite(PROJECT_PATH + "Output/descriptor_match/" + std::to_string(result1.filetype.number) + "-" + std::to_string(result2.filetype.number) + ".png", img_matches);

    if(DEBUG)
      std::cout << "Estimating fundamental matrix..." << std::endl;
    cv::Mat fundamental_matrix;
    find_fundamental_matrix(result1.keypoints, result2.keypoints, good_matches, fundamental_matrix);

    if(DEBUG)
      std::cout << "Rectifying images..." << std::endl;
    cv::Mat left_rectified, right_rectified;
    rectify_images(left, right, result1.keypoints, result2.keypoints, good_matches, fundamental_matrix, camParams, left_rectified, right_rectified);
    imwrite(PROJECT_PATH + "Output/left_rectified/left-" + img1_name, left_rectified);
    imwrite(PROJECT_PATH + "Output/right_rectified/right-" + img2_name, right_rectified);

    // left_rectified = left;
    // right_rectified = right;

    if(DEBUG)
      std::cout << "Computing disparity map..." << std::endl;
    // cv::Mat disp = cv::Mat(left_rectified.rows, left_rectified.cols, CV_16U);
    cv::Mat disp, disp_vis;
    compute_disparity_map_modified(left_rectified, right_rectified, disp, disp_vis);
    // compute_disparity_map(left_rectified, right_rectified, disp);
    // imwrite(PROJECT_PATH + "Output/disparity_map/" + img1_name, disp);
    // cv::Mat disp_vis;
    // cv::ximgproc::getDisparityVis(disp, disp_vis);
    imwrite(PROJECT_PATH + "Output/disparity_map/" + img1_name, disp);

    if(USE_REPROJECT)
    {
      if(DEBUG)
	std::cout << "Reprojecting disparity map and generating point cloud..." << std::endl;
      reproject_to_3d(disp, left_rectified, camParams);
    }
    else {
      std::cout << "Without using reproject to 3d" << std::endl;
      if(DEBUG)
	std::cout << "Computing depth map and generating point cloud..." << std::endl;
      cv::Mat depthMap = cv::Mat::zeros(disp.rows, disp.cols, CV_64F);
      get_depth_map_from_disparity_map(disp, camParams, depthMap);
      get_point_cloud_from_depth_map(depthMap, left_rectified, camParams, std::to_string(result1.filetype.number));
    }
    std::cout << "Processed " << filename1 << " and " << filename2 << std::endl
              << std::endl;
    return;
}

int main()
{
    print_running_information();
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
	    std::string color_intrinsics = dataset_dir + "/colorIntrinsics.txt";
	    std::string depth_intrinsics = dataset_dir + "/depthIntrinsics.txt";
            while ((entry = readdir(directory)) != NULL)
            {
                if (entry->d_name[0] != 'f')
                    continue;
                struct filenameType filename_type = extract_file_name(entry->d_name);
                if (filename_type.category != 0)
                    continue;
		struct cameraParams camParams;
		std::string left_pose = dataset_dir + "/" + get_file_name(filename_type.number, 2);
		std::string right_pose = dataset_dir + "/" + get_file_name(filename_type.number + 1, 2);
		// TODO: add baseline calculation of every two images
		getCameraParamsBricks(color_intrinsics, depth_intrinsics, left_pose, right_pose, &camParams);
                process_pair_images(dataset_dir + "/" + entry->d_name, dataset_dir + "/" + get_file_name(filename_type.number + 1, filename_type.category), camParams);
                if (TEST)
                    break;
            }
        }
        closedir(directory);
    }
    else if (compare_string(DATASET, "kitti"))
    {
        std::string left_dir = dataset_dir + "/data_scene_flow/training/image_2/";
        std::string right_dir = dataset_dir + "/data_scene_flow/training/image_3/";
        std::string calib_dir = dataset_dir + "/data_scene_flow_calib/training/calib_cam_to_cam/";
        std::string disp_dir = dataset_dir + "/data_scene_flow/training/disp_occ_0/";
        DIR *directory = opendir(left_dir.c_str());
        struct dirent *entry;
        if (directory == NULL)
        {
            std::cout << "Error in main.cpp main(): Directory not found or failed to open directory." << left_dir << std::endl;
            return -1;
        }
        else
        {
            while ((entry = readdir(directory)) != NULL)
            {
                if (entry->d_name[0] == '.')
                    continue;
                std::string filename = entry->d_name;
                std::string calib_file = calib_dir + filename.substr(filename.find_last_of("/") + 1, filename.find("_")) + ".txt";
                struct cameraParams camParams;
                getCameraParamsKITTI(calib_file, &camParams);
                process_pair_images(left_dir + filename, right_dir + filename, camParams);
                if (TEST)
                    break;
            }
        }
        closedir(directory);
    }
    if (RECONSTRUCT)
        merge(MODELS_DIR);
    if (compare_string(DATASET, "kitti") && COMPARE_DENSE_MATCHING)
    {
        std::string disp_dir = PROJECT_PATH + "Output/disparity_map/";
        std::string groundtruth_disp_dir = dataset_dir + "/data_scene_flow/training/disp_occ_0/";
        compute_disparity_performance(disp_dir, groundtruth_disp_dir);
    }
    std::cout << "Stereo Reconstruction Finished" << std::endl;
    return 0;
}
