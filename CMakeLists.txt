cmake_minimum_required(VERSION 3.6)

project(StereoReconstruction)

set(CMAKE_CXX_STANDARD 14)

set(JPEG_LIB_VERSION 90)

set(freeimage_DIR "/opt/homebrew/Cellar/freeimage/3.18.0/include")

find_package(Eigen3 CONFIG REQUIRED)
find_package(Flann CONFIG REQUIRED)
find_package(Ceres CONFIG REQUIRED)
find_package(OpenCV CONFIG REQUIRED)

add_executable(StereoReconstruction main.cpp ${HEADER_FILES} ${SOURCE_FILES})
target_include_directories(StereoReconstruction PUBLIC ${freeimage_DIR})
target_link_libraries(StereoReconstruction PRIVATE Eigen3::Eigen)
target_link_libraries(StereoReconstruction PRIVATE flann::flann_s flann::flann_cpp_s)
target_link_libraries(StereoReconstruction PRIVATE ceres Ceres::ceres)
target_link_libraries(StereoReconstruction PRIVATE opencv_ml tegra_hal opencv_dnn opencv_core)
target_link_libraries(StereoReconstruction PRIVATE opencv_imgcodecs opencv_highgui opencv_xfeatures2d opencv_ximgproc)
target_link_libraries(StereoReconstruction PRIVATE opencv_imgproc opencv_features2d opencv_calib3d opencv_flann)

# cmake -B . -S . -DCMAKE_TOOLCHAIN_FILE=./vcpkg/scripts/buildsystems/vcpkg.cmake
# cmake --build .
