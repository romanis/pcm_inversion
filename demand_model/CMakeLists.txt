project (pcm_kernel)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(OpenMP REQUIRED)

# The PCM kernel (docs/estimation_harness/DESIGN.md 3.0): layer 0 pcm_geometry (market_share/), layer 1
# pcm_aggregate, layer 2 pcm_invert, the functionals pcm_welfare, pcm_second_choice, pcm_groups, pcm_events and the
# facade pcm_kernel. Static and position independent so the Python module nfp_harness/_native/pcm_kernel is
# self-contained (DESIGN 1.2).
add_library(pcm_kernel STATIC
    pcm_kernel.cpp pcm_aggregate.cpp pcm_invert.cpp pcm_events.cpp pcm_welfare.cpp pcm_groups.cpp
    pcm_second_choice.cpp ${ROOT}/market_share/pcm_geometry.cpp ${ROOT}/market_share/pcm_market_share.cpp)
set_target_properties(pcm_kernel PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(pcm_kernel PUBLIC ${ROOT}/market_share ${ROOT}/demand_model ${Boost_INCLUDE_DIR})
target_link_libraries(pcm_kernel PUBLIC Eigen3::Eigen OpenMP::OpenMP_CXX)

enable_testing()
include(GoogleTest)
foreach(t pcm_aggregate_test pcm_kernel_test pcm_events_test pcm_welfare_test pcm_truncation_test pcm_groups_test pcm_second_choice_test)
  add_executable(${t} ${t}.cpp)
  target_link_libraries(${t} gtest_main pcm_kernel)
  gtest_discover_tests(${t} TEST_PREFIX "${t}.")   # names match ctest -R "pcm_(aggregate|kernel|events|welfare|truncation|groups|second_choice)"
endforeach()
