project (share_computation)
set(CMAKE_BUILD_TYPE Release)

# Only uncomment it if you know what you are doing. It typically not work due to different openmp flag for different compilers
# if(APPLE)
# if(CMAKE_C_COMPILER_ID MATCHES "Clang")
#     set(OpenMP_C "${CMAKE_C_COMPILER}" CACHE STRING "" FORCE)
#     set(OpenMP_C_FLAGS "-fopenmp=libomp -Wno-unused-command-line-argument" CACHE STRING "" FORCE)
#     set(OpenMP_C_LIB_NAMES "libomp" "libgomp" "libiomp5" CACHE STRING "" FORCE)
#     set(OpenMP_libomp_LIBRARY ${OpenMP_C_LIB_NAMES} CACHE STRING "" FORCE)
#     set(OpenMP_libgomp_LIBRARY ${OpenMP_C_LIB_NAMES} CACHE STRING "" FORCE)
#     set(OpenMP_libiomp5_LIBRARY ${OpenMP_C_LIB_NAMES} CACHE STRING "" FORCE)
# endif()
# if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
#   set(OpenMP_CXX "${CMAKE_CXX_COMPILER}" CACHE STRING "" FORCE)
#   set(OpenMP_CXX_FLAGS "-fopenmp=libomp -Wno-unused-command-line-argument" CACHE STRING "" FORCE)
#   set(OpenMP_CXX_LIB_NAMES "libomp" "libgomp" "libiomp5" CACHE STRING "" FORCE)
#   set(OpenMP_libomp_LIBRARY ${OpenMP_CXX_LIB_NAMES} CACHE STRING "" FORCE)
#   set(OpenMP_libgomp_LIBRARY ${OpenMP_CXX_LIB_NAMES} CACHE STRING "" FORCE)
#   set(OpenMP_libiomp5_LIBRARY ${OpenMP_CXX_LIB_NAMES} CACHE STRING "" FORCE)
# endif()
# endif()


find_package(Boost REQUIRED)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
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