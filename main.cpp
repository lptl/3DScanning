#include <iostream>
#include <fstream>
#include <array>
#include <dirent.h>

#include "Libs/Pipeline.h"


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
    std::cout << "Computing disparity map for image " << std::endl;
    compute_disparity_map(img1, img2);
    std::cout << "Finished processing " << filename1 << " and " << filename2 << std::endl;
    return;
}

int main()
{
    std::string dataset_dir = PROJECT_PATH + "bricks-rgbd/";
    DIR* directory = opendir(dataset_dir.c_str());
    struct dirent* entry;
    if(directory == NULL){
        std::cout << "Error in main.cpp main(): Directory not found or failed to open directory." << std::endl;
        return -1;
    } else {
        while((entry = readdir(directory)) != NULL){
            if(entry->d_name[0] != 'f')
                continue;
            struct filenameType filename_type = extract_file_name(entry->d_name);
            if(filename_type.category != 0)
                continue;
            process_pair_images(dataset_dir, entry->d_name, get_file_name(filename_type.number+1, filename_type.category));
        }
        closedir(directory);
    }
    std::cout << "Stereo Reconstruction Finished" << std::endl;
    return 0;
}