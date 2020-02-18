cmake_minimum_required(VERSION 3.15)
project(test_cpp)

set(CMAKE_CXX_STANDARD 14)
find_package(autodiff REQUIRED)

add_subdirectory(computing)