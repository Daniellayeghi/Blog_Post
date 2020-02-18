add_executable(test autodiff_test.cpp)
include_directories(test ${PROJECT_SOURCE_DIR}/utils)
target_link_libraries(test autodiff::autodiff ${PROJECT_SOURCE_DIR}/utils/time_bench.h)