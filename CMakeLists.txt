cmake_minimum_required(VERSION 3.10)

project (PCM_inversion)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(market_share)

find_package (Tasmanian 7.7.0 PATHS "/usr/local/TASMANIAN" REQUIRED  SHARED OPENMP BLAS)

set(Eigen3_DIR "/usr/share/eigen3/cmake/")
find_package (Eigen3 REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/market_share)


# add_library(test SHARED ./pcm_market_share.cpp)
# set_target_properties(test PROPERTIES PUBLIC_HEADER ./pcm_market_share.hpp)

# target_link_libraries(test Tasmanian::Tasmanian)
# target_link_libraries(test Eigen3::Eigen)

# add the executable
# find_package (Eigen3 3.4 REQUIRED NO_MODULE)
add_executable(test_pcm test_pcm.cpp)
find_package("NLopt")
target_link_directories(test_pcm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_pcm Eigen3::Eigen pcm_market_share Tasmanian::Tasmanian NLopt::nlopt)
# set_target_properties(PCM_inversion PROPERTIES OUTPUT_NAME test_pcm)