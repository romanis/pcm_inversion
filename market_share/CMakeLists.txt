project (share_computation)
set(CMAKE_BUILD_TYPE Release)


find_package(Boost REQUIRED)
find_package (Eigen3 3.4 REQUIRED NO_MODULE)


add_library(pcm_market_share SHARED ./pcm_market_share.cpp)
target_include_directories(pcm_market_share PUBLIC ${Boost_INCLUDE_DIR} )

set_target_properties(pcm_market_share PROPERTIES PUBLIC_HEADER ./pcm_market_share.hpp)
set_target_properties(pcm_market_share PROPERTIES PUBLIC_HEADER "${ROOT}/market_share/pcm_market_share.hpp")


target_link_libraries(pcm_market_share Eigen3::Eigen )
target_link_libraries(pcm_market_share ${Boost_LIBRARIES})