cmake_minimum_required(VERSION 3.26)
project(channel_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type for Channel Native." FORCE)
endif()

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path)"
    RESULT_VARIABLE TORCH_CMAKE_PREFIX_RESULT
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT TORCH_CMAKE_PREFIX_RESULT EQUAL 0)
    message(FATAL_ERROR "Could not locate PyTorch CMake prefix. Install torch in the build environment.")
endif()
list(PREPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX}")
find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib" REQUIRED)

Python_add_library(
    _channel_native
    MODULE
    WITH_SOABI
    native/channel_native/bindings.cpp
    native/channel_native/build_info.cpp
    native/channel_native/path_solver.cpp
)

target_compile_features(_channel_native PRIVATE cxx_std_17)
target_link_libraries(_channel_native PRIVATE "${TORCH_LIBRARIES}" "${TORCH_PYTHON_LIBRARY}")

install(TARGETS _channel_native LIBRARY DESTINATION witwin/channel_native)
