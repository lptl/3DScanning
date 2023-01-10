#include <iostream>
#include <fstream>
#include <array>
#include <dirent.h>

#include "Libs/Pipeline.h"

// this is the directory to store small chunks of models
#define MODELS_DIR "Models/"


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
    // 1. use rectified images to do stereo matching
    // https://docs.opencv.org/3.4/d2/d6e/classcv_1_1StereoMatcher.html#a03f7087df1b2c618462eb98898841345
    // 2.reconstruct small chunks 3D model from disparity map
    // https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga4b1dab0d1d0d0f3a67b621a27c39b5a5
    std::cout << "Computing disparity map for image " << std::endl;
    compute_disparity_map(img1, img2);
    std::cout << "Finished processing " << filename1 << " and " << filename2 << std::endl;
    return;
}

void reconstruct(std::string models_directory){
    DIR* directory = opendir(models_directory.c_str());
    struct dirent* entry;
    int index = 0;
    std::string base_model, target_model, other_model;
    while(directory != NULL){
        while((entry = readdir(directory)) != NULL){
            if(index == 0){
                base_model = models_directory + std::string(entry->d_name);
                index++;
                continue;
            }
            other_model = models_directory + std::string(entry->d_name);
            target_model = models_directory + std::to_string(index) + ".off";
            if(!icp_reconstruct(base_model, other_model, target_model)){
                std::cout << "Error: Failed to reconstruct model. Skipped one model: " + base_model << std::endl;
                base_model = other_model;
            }
            else base_model = target_model;
            index++;
        }   
    }
}

int main()
{
    std::string dataset_dir = PROJECT_PATH + "bricks-rgbd/";
    DIR* directory = opendir(dataset_dir.c_str());
    /*
    Compute the running time of function
    computeRunTime([&](){std::vector<detectResult> results = processXimages(20, dataset_dir);});
     */
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
        reconstruct(MODELS_DIR);
    }
    std::cout << "Stereo Reconstruction Finished" << std::endl;
    return 0;
}