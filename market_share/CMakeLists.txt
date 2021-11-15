project (share_computation)
set(CMAKE_BUILD_TYPE Release)


find_package (Tasmanian 7.7.0 PATHS "/usr/local/TASMANIAN" REQUIRED  SHARED OPENMP BLAS CUDA MPI)
find_package (Eigen3 3.4 REQUIRED NO_MODULE)


add_library(pcm_market_share SHARED ./pcm_market_share.cpp)
set_target_properties(pcm_market_share PROPERTIES PUBLIC_HEADER ./pcm_market_share.hpp)

target_link_libraries(pcm_market_share Tasmanian::Tasmanian)
target_link_libraries(pcm_market_share Eigen3::Eigen)