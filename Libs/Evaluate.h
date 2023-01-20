#pragma once
#include <iostream>
#include <dirent.h>

#include "Libs/Pipeline.h"

template<typename Func>
double computeRunTime(Func f){

    int64 start = cv::getTickCount();

    f();

    int64 end = cv::getTickCount();
    double runtime = (end - start) / cv::getTickFrequency();
    std::cout << "Runtime: " << runtime << " seconds" << std::endl;
    std::cout << "Stereo Reconstruction Finished" << std::endl;
    return runtime;
}

std::vector<struct detectResult> processXimages(int image_number,std::string dataset_dir){
    DIR* directory = opendir(dataset_dir.c_str());
    struct dirent* entry;
    std::vector<struct detectResult> reusltVector;
    int readCount = 0;
    if(directory == NULL){
        std::cout << "Can not read file from the dataset directory" << std::endl;
        return reusltVector;
    } else {
        while((entry = readdir(directory)) != NULL && readCount < image_number){
            if(entry->d_name[0] != 'f')
                continue;
            struct filenameType filename_type = extract_file_name(entry->d_name);
            if(filename_type.category != 0)
                continue;
            std::string filename = get_file_name(filename_type.number, filename_type.category);
            std::cout << "Processing image " << filename << std::endl;
            Mat img = imread(dataset_dir + filename, IMREAD_COLOR);
            struct detectResult result = detect_keypoints_or_features(dataset_dir, filename, img);
            reusltVector.push_back(result);
            readCount++;
        }
        closedir(directory);
    }
    return reusltVector;
}