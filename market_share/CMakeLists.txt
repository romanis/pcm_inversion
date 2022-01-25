project (share_computation)
set(CMAKE_BUILD_TYPE Release)


find_package(Boost REQUIRED)
find_package (Eigen3 3.4 REQUIRED NO_MODULE)
find_package(OpenMP)

add_library(pcm_market_share SHARED ./pcm_market_share.cpp)
target_include_directories(pcm_market_share PUBLIC ${Boost_INCLUDE_DIR} )

set_target_properties(pcm_market_share PROPERTIES PUBLIC_HEADER ./pcm_market_share.hpp)
set_target_properties(pcm_market_share PROPERTIES PUBLIC_HEADER "${ROOT}/market_share/pcm_market_share.hpp")


target_link_libraries(pcm_market_share PUBLIC Eigen3::Eigen )
target_link_libraries(pcm_market_share PUBLIC ${Boost_LIBRARIES})
if(OpenMP_CXX_FOUND)
    target_link_libraries(pcm_market_share PUBLIC OpenMP::OpenMP_CXX)
endif()


enable_testing()

add_executable(
    market_share_test
    market_share_test.cpp
)

target_include_directories(market_share_test PUBLIC ${Boost_INCLUDE_DIR} )
target_include_directories(market_share_test PUBLIC ${Eigen3_INCLUDE_DIR} )

target_link_libraries(
    market_share_test
    gtest_main
    Eigen3::Eigen
    ${Boost_LIBRARIES}
    pcm_market_share
)

include(GoogleTest)
gtest_discover_tests(market_share_test)