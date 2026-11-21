cmake_minimum_required(VERSION 3.18)
add_compile_options(-Wall)
project (PCM_inversion)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
# Release keeps the -O3 code and adds full debug info, so a crash in a long run leaves a core dump
# with symbols, source lines and stacks (gdb unwinds -O3 frames from the DWARF call-frame info).
# -g does not change the generated code. Frame pointers are not added: they cost 3.4% on the share
# kernel (pcm_events perf test, 16 threads: 0.0474 s -> 0.0490 s).
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -g" CACHE STRING "" FORCE)
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_CXX_STANDARD 17)

include_directories(${ROOT})

# Lets ctest run every subdirectory's gtests from the top of the build tree.
enable_testing()

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/release-1.11.0.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
FetchContent_MakeAvailable(googletest)

add_subdirectory(market_share)
add_subdirectory(inversion_algorithm)
add_subdirectory(demand_model)

# FIND_PACKAGE(LATEX)
# IF(LATEX_FOUND)
#     add_subdirectory(latex_article)
# ENDIF()

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

install(TARGETS market_inversion pcm_market_share python_market_share python_pcm_inversion
    LIBRARY DESTINATION "${CMAKE_INSTALL_PREFIX}/lib"
    PUBLIC_HEADER DESTINATION "${CMAKE_INSTALL_PREFIX}/include")