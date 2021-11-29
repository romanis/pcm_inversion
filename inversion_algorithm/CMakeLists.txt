project (market_inversion)
set(CMAKE_BUILD_TYPE Release)

find_package (Eigen3 3.4 REQUIRED NO_MODULE)
find_package("NLopt")

add_library(market_inversion SHARED ./market_inversion.cpp)
set_target_properties(market_inversion PROPERTIES PUBLIC_HEADER ./market_inversion.hpp)

message("adding target include directory ${ROOT}/market_share")

target_include_directories(market_inversion PUBLIC ${ROOT}/market_share)

target_link_libraries(market_inversion Eigen3::Eigen)
target_link_libraries(market_inversion pcm_market_share)
target_link_libraries(market_inversion NLopt::nlopt)