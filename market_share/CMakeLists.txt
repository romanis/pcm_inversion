project (share_computation LANGUAGES CXX)
set(CMAKE_BUILD_TYPE Release)
cmake_minimum_required(VERSION 3.18)
# set compiler to homebrew gcc and g++
# set(CMAKE_C_COMPILER "/usr/local/bin/gcc-11")


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
find_package (Eigen3 3.4 REQUIRED NO_MODULE)
find_package(OpenMP)
# find the pybind to make python binding
# set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)
find_package(Python 3.7 COMPONENTS Interpreter Development REQUIRED)

add_library(pcm_market_share SHARED ./pcm_market_share.cpp)
target_include_directories(pcm_market_share PUBLIC ${Boost_INCLUDE_DIR} )

# this is needed bencase python binding stuff is in .cpp file that impelements the library
target_include_directories(pcm_market_share PUBLIC ${Python_INCLUDE_DIRS}) 

# set_target_properties(pcm_market_share PROPERTIES PUBLIC_HEADER ./pcm_market_share.hpp)
set_target_properties(pcm_market_share PROPERTIES PUBLIC_HEADER "${ROOT}/market_share/pcm_market_share.hpp")


target_link_libraries(pcm_market_share PUBLIC Eigen3::Eigen )
target_link_libraries(pcm_market_share PUBLIC ${Boost_LIBRARIES})
if(OpenMP_CXX_FOUND)
    target_link_libraries(pcm_market_share PUBLIC OpenMP::OpenMP_CXX)
endif()

# create python library if pybind11 is found
if(pybind11_FOUND)

    pybind11_add_module(python_market_share ./pcm_market_share.cpp)
    target_include_directories(python_market_share PUBLIC ${Boost_INCLUDE_DIR} )
    target_include_directories(python_market_share PUBLIC ${Python_INCLUDE_DIRS})
    message(STATUS "Python include directories: ${Python_INCLUDE_DIRS}")

    # print include directories of python_market_share

    set_target_properties(python_market_share PROPERTIES PUBLIC_HEADER "${ROOT}/market_share/pcm_market_share.hpp")
    
    target_link_libraries(python_market_share PUBLIC Eigen3::Eigen )
    target_link_libraries(python_market_share PUBLIC ${Boost_LIBRARIES})
    target_link_libraries(python_market_share PUBLIC pybind11::module)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(python_market_share PUBLIC OpenMP::OpenMP_CXX)
    endif()
endif()



# create testing
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