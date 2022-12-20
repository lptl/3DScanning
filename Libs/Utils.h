#pragma once

#include <string>
#include <vector>
#include "Types.h"

struct filenameType extract_file_name(std::string filename){
    filenameType type;
    type.name = filename;
    type.number = std::stoi(filename.substr(9, 12));
    if(filename.find("color") != std::string::npos)
        type.category = 0;
    else if(filename.find("depth") != std::string::npos)
        type.category = 1;
    else if(filename.find("pose") != std::string::npos)
        type.category = 2;
    else
        type.category = -1;
    return type;
}

std::string get_file_name(int number, int category){
    if(number >= 772)
        number = 770;
    std::string number_string = std::to_string(number);
    if(number < 10)
        number_string = "00" + number_string;
    else if(number < 100)
        number_string = "0" + number_string;
    std::string filename = "frame-000" + number_string + ".";
    if(category == 0)
        filename += "color.png";
    else if(category == 1)
        filename += "depth.png";
    else if(category == 2)
        filename += "pose.txt";
    return filename;
}

bool compare_string(std::string str1, std::string str2){
    if(str1.compare(str2) == 0)
        return true;
    else
        return false;
}