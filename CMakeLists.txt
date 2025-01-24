cmake_minimum_required(VERSION 3.18)
add_compile_options(-Wall)
project (PCM_inversion)
set(CMAKE_BUILD_TYPE Release)
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_CXX_STANDARD 17)

include_directories(${ROOT})

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/release-1.11.0.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
FetchContent_MakeAvailable(googletest)

add_subdirectory(market_share)
add_subdirectory(inversion_algorithm)

find_package(pybind11 CONFIG REQUIRED)
find_package(Python 3.7 COMPONENTS Interpreter Development REQUIRED)
if(Python_FOUND AND pybind11_FOUND)
    message("found python and pybind11, adding python bindings")
    add_subdirectory(python_bindings)
endif()


find_package(Tasmanian)
IF(Tasmanian_FOUND)
    message("found Tasmanian, building examples")
    add_subdirectory(examples)
ENDIF()

message("install lib ${CMAKE_INSTALL_LIBDIR}, include ${CMAKE_INSTALL_INCLUDEDIR}")

install(TARGETS market_inversion pcm_market_share python_market_share
    LIBRARY DESTINATION "${CMAKE_INSTALL_PREFIX}/lib"
    PUBLIC_HEADER DESTINATION "${CMAKE_INSTALL_PREFIX}/include")