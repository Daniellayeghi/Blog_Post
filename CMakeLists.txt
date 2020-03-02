cmake_minimum_required(VERSION 3.15)
project(test_cpp)

set(CMAKE_CXX_STANDARD 17)
find_package(autodiff REQUIRED)
find_package(Eigen3 REQUIRED)


add_subdirectory(computing)
add_subdirectory(utils)