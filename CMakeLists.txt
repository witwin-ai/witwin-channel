cmake_minimum_required(VERSION 3.26)

set(CHANNEL_NATIVE_DEFAULT_CUDA_ARCHITECTURES "75-real;80-real;86-real;89-real;120-real;120-virtual")
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "" OR CMAKE_CUDA_ARCHITECTURES STREQUAL "52")
    set(CMAKE_CUDA_ARCHITECTURES "${CHANNEL_NATIVE_DEFAULT_CUDA_ARCHITECTURES}" CACHE STRING "CUDA architectures for Channel Native CUDA targets." FORCE)
endif()
set(CHANNEL_NATIVE_DEFAULT_TORCH_CUDA_ARCH_LIST "7.5 8.0 8.6 8.9 12.0+PTX")
if("$ENV{TORCH_CUDA_ARCH_LIST}" STREQUAL "")
    set(ENV{TORCH_CUDA_ARCH_LIST} "${CHANNEL_NATIVE_DEFAULT_TORCH_CUDA_ARCH_LIST}")
endif()

project(channel_native LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type for Channel Native." FORCE)
endif()

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

option(CHANNEL_NATIVE_BUILD_RAYDN "Build vendored RayDN native backend." ON)

if(CHANNEL_NATIVE_BUILD_RAYDN)
    set(RAYDN_BUILD_NATIVE ON CACHE BOOL "Build RayDN native extension for Channel Native." FORCE)
    add_subdirectory(ext/raydn)
endif()

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
    native/channel_native/kernels/accum.cu
    native/channel_native/kernels/diffraction.cu
    native/channel_native/kernels/deterministic_accum.cu
    native/channel_native/kernels/deterministic_field.cu
    native/channel_native/kernels/los.cu
    native/channel_native/kernels/material.cu
    native/channel_native/kernels/reflection.cu
    native/channel_native/kernels/sampling.cu
    native/channel_native/montecarlo_basic.cpp
    native/channel_native/path_solver.cpp
)

target_compile_features(_channel_native PRIVATE cxx_std_17)
target_link_libraries(_channel_native PRIVATE "${TORCH_LIBRARIES}" "${TORCH_PYTHON_LIBRARY}")
if(TARGET _raydn)
    add_dependencies(_channel_native _raydn)
    target_compile_definitions(_channel_native PRIVATE CHANNEL_NATIVE_WITH_RAYDN=1 CHANNEL_NATIVE_WITH_OPTIX=1)
else()
    target_compile_definitions(_channel_native PRIVATE CHANNEL_NATIVE_WITH_RAYDN=0 CHANNEL_NATIVE_WITH_OPTIX=0)
endif()

install(TARGETS _channel_native LIBRARY DESTINATION witwin/channel_native)
if(TARGET _raydn)
    install(TARGETS _raydn LIBRARY DESTINATION witwin/channel_native/_native RUNTIME DESTINATION witwin/channel_native/_native)
endif()
