cmake_minimum_required(VERSION 3.6)

project(StereoReconstruction)

# Set C++ flags
set(CMAKE_CXX_STANDARD 14)

set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain file")

# execute the following command to install libraries
# ./vcpkg/vcpkg install ceres eigen3 flann opencv opencv[contrib]
# set(OpenCV_DIR "${VCPKG_INSTALLED_DIR}/arm64-osx/share/opencv4")
set(JPEG_LIB_VERSION 90)

find_package(Eigen3 CONFIG REQUIRED)
find_package(Flann CONFIG REQUIRED)
find_package(Ceres CONFIG REQUIRED)
find_package(OpenCV CONFIG REQUIRED)

# link libraries
add_executable(StereoReconstruction main.cpp)
target_link_libraries(StereoReconstruction PRIVATE Eigen3::Eigen)
target_link_libraries(StereoReconstruction PRIVATE flann::flann_s flann::flann_cpp_s)
target_link_libraries(StereoReconstruction PRIVATE ceres Ceres::ceres)
target_link_libraries(StereoReconstruction PRIVATE opencv_ml tegra_hal opencv_dnn opencv_core)
target_link_libraries(StereoReconstruction PRIVATE opencv_imgcodecs opencv_highgui opencv_xfeatures2d opencv_ximgproc)
target_link_libraries(StereoReconstruction PRIVATE opencv_imgproc opencv_features2d opencv_calib3d opencv_flann)