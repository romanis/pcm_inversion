add_executable(test_pcm test_pcm.cpp)
find_package (Eigen3 3.4 REQUIRED NO_MODULE)
find_package("NLopt")
find_package (Tasmanian REQUIRED  SHARED OPENMP BLAS CUDA MPI)


target_include_directories(test_pcm PUBLIC "${ROOT}/inversion_algorithm")

target_link_directories(test_pcm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_pcm Eigen3::Eigen Tasmanian::Tasmanian NLopt::nlopt market_inversion pcm_market_share)
