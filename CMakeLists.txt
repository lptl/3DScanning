cmake_minimum_required(VERSION 3.6)

project(StereoReconstruction)

set(CMAKE_CXX_STANDARD 14)

set(JPEG_LIB_VERSION 90)

find_package(Eigen3 CONFIG REQUIRED)
find_package(Flann CONFIG REQUIRED)
find_package(Ceres CONFIG REQUIRED)
find_package(freeimage CONFIG REQUIRED)
find_package(PNG REQUIRED)
find_package(OpenCV CONFIG REQUIRED)

FILE(GLOB MyLibs *.h)

set(HEADER_FILES 
    ${MyLibs}
)

set(SOURCE_FILES 
    Libs/FreeImageHelper.cpp
)

add_executable(StereoReconstruction main.cpp ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(StereoReconstruction PRIVATE Eigen3::Eigen)
target_link_libraries(StereoReconstruction PRIVATE flann::flann flann::flann_cpp)
target_link_libraries(StereoReconstruction PRIVATE ceres Ceres::ceres)
target_link_libraries(StereoReconstruction PRIVATE freeimage::FreeImage freeimage::FreeImagePlus)
target_link_libraries(StereoReconstruction PRIVATE ${OpenCV_LIBS})