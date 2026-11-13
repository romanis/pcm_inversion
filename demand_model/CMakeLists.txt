project (pcm_demand)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Eigen3 3.4 REQUIRED NO_MODULE)

add_library(pcm_demand SHARED ./pcm_demand.cpp)
set_target_properties(pcm_demand PROPERTIES PUBLIC_HEADER "${ROOT}/demand_model/pcm_demand.hpp")
target_include_directories(pcm_demand PUBLIC ${ROOT}/market_share ${ROOT}/demand_model)
target_link_libraries(pcm_demand PUBLIC Eigen3::Eigen pcm_market_share)

enable_testing()
add_executable(pcm_demand_test pcm_demand_test.cpp)
target_link_libraries(pcm_demand_test gtest_main Eigen3::Eigen pcm_demand)
include(GoogleTest)
gtest_discover_tests(pcm_demand_test)
