#add_executable(test autodiff_test.cpp autodiff_test_2.cpp)
#include_directories(test ${PROJECT_SOURCE_DIR}/utils)
#target_link_libraries(test autodiff::autodiff ${PROJECT_SOURCE_DIR}/utils/time_bench.h)
set(HEADER_LIST autodiff_test_2.h)
set(SOURCE_LIST autodiff_test_2.cpp)
add_library(test_lib ${SOURCE_LIST} ${HEADER_LIST})
link_libraries(test_lib autodiff::autodiff Eigen3::Eigen)
add_executable(test_2 autodiff_test.cpp)
include_directories(test_2 ${PROJECT_SOURCE_DIR}/utils)
target_link_libraries(test_2 autodiff::autodiff  test_lib ${PROJECT_SOURCE_DIR}/utils/time_bench.h)