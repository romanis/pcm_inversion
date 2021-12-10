cmake_minimum_required(VERSION 3.10)

project (PCM_inversion)
set(CMAKE_BUILD_TYPE Release)
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(ROOT ${CMAKE_CURRENT_SOURCE_DIR})

include_directories(${ROOT})

add_subdirectory(market_share)
add_subdirectory(inversion_algorithm)
find_package(Tasmanian)
IF(Tasmanian_FOUND)
    message("found Tasmanian, building examples")
    add_subdirectory(examples)
ENDIF()

message("install lib ${CMAKE_INSTALL_LIBDIR}, include ${CMAKE_INSTALL_INCLUDEDIR}")

install(TARGETS market_inversion pcm_market_share
    LIBRARY DESTINATION /usr/local/lib
    PUBLIC_HEADER DESTINATION /usr/local/include)