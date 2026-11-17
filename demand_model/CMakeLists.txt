project (pcm_demand)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(OpenMP REQUIRED)

add_library(pcm_demand SHARED ./pcm_demand.cpp)
set_target_properties(pcm_demand PROPERTIES PUBLIC_HEADER "${ROOT}/demand_model/pcm_demand.hpp")
target_include_directories(pcm_demand PUBLIC ${ROOT}/market_share ${ROOT}/demand_model)
target_link_libraries(pcm_demand PUBLIC Eigen3::Eigen pcm_market_share OpenMP::OpenMP_CXX)

enable_testing()
add_executable(pcm_demand_test pcm_demand_test.cpp)
target_link_libraries(pcm_demand_test gtest_main Eigen3::Eigen pcm_demand)
include(GoogleTest)
gtest_discover_tests(pcm_demand_test)

# Estimation harness kernel (docs/estimation_harness/DESIGN.md 1.2). Static and position independent
# so the Python module nfp_harness/_native/pcm_kernel is self-contained.
add_library(pcm_kernel STATIC
    pcm_kernel.cpp pcm_events.cpp pcm_welfare.cpp pcm_demand.cpp
    ${ROOT}/market_share/pcm_market_share.cpp)
set_target_properties(pcm_kernel PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(pcm_kernel PUBLIC ${ROOT}/market_share ${ROOT}/demand_model ${Boost_INCLUDE_DIR})
target_link_libraries(pcm_kernel PUBLIC Eigen3::Eigen OpenMP::OpenMP_CXX)
foreach(t pcm_kernel_test pcm_events_test pcm_welfare_test pcm_truncation_test)
  add_executable(${t} ${t}.cpp)
  target_link_libraries(${t} gtest_main pcm_kernel)
  gtest_discover_tests(${t} TEST_PREFIX "${t}.")   # names match ctest -R "pcm_(kernel|events|welfare|truncation)"
endforeach()
