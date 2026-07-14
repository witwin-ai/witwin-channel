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

set(
    RAYD_SOURCE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../../RayDi"
    CACHE PATH
    "Path to the RayD source tree used by Channel Native.")
get_filename_component(RAYD_SOURCE_DIR "${RAYD_SOURCE_DIR}" ABSOLUTE)
if(NOT EXISTS "${RAYD_SOURCE_DIR}/backends/torch/CMakeLists.txt")
    message(FATAL_ERROR
        "RAYD_SOURCE_DIR='${RAYD_SOURCE_DIR}' is not a RayD source tree. "
        "Set -DRAYD_SOURCE_DIR=<path-to-RayDi>.")
endif()
set(RAYD_TORCH_BUILD_NATIVE ON CACHE BOOL "Build RayD Torch native core." FORCE)
set(RAYD_TORCH_BUILD_PYTHON_MODULE OFF CACHE BOOL "Do not build RayD's Python module inside Channel Native." FORCE)
add_subdirectory(
    "${RAYD_SOURCE_DIR}/backends/torch"
    "${CMAKE_CURRENT_BINARY_DIR}/rayd-torch")
if(NOT TARGET rayd_torch_native_core)
    message(FATAL_ERROR "RayD did not provide the rayd_torch_native_core target.")
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
if(WIN32 AND NOT EXISTS "${TORCH_CMAKE_PREFIX}/Torch/TorchConfig.cmake")
    get_filename_component(PYTHON_ENV_ROOT "${Python_EXECUTABLE}" DIRECTORY)
    list(PREPEND CMAKE_PREFIX_PATH
        "${PYTHON_ENV_ROOT}/Library"
        "${PYTHON_ENV_ROOT}/Lib/site-packages")
    set(CONDA_TORCH_DIR "${PYTHON_ENV_ROOT}/Library/share/cmake/Torch")
    if(EXISTS "${CONDA_TORCH_DIR}/TorchConfig.cmake")
        set(Torch_DIR "${CONDA_TORCH_DIR}")
    endif()
endif()
find_package(Torch REQUIRED)
get_filename_component(TORCH_PACKAGE_DIR "${TORCH_CMAKE_PREFIX}/../.." ABSOLUTE)
find_library(
    TORCH_PYTHON_LIBRARY
    torch_python
    PATHS
        "${TORCH_INSTALL_PREFIX}/lib"
        "${TORCH_PACKAGE_DIR}/lib"
    REQUIRED)

Python_add_library(
    _channel_native
    MODULE
    WITH_SOABI
    native/channel_native/bindings.cpp
    native/channel_native/build_info.cpp
    native/channel_native/bdpt.cpp
    native/channel_native/raydn_bridge.cpp
    native/channel_native/kernels/accum.cu
    native/channel_native/kernels/bdpt_accum.cu
    native/channel_native/kernels/bdpt_connect.cu
    native/channel_native/kernels/bdpt_sampling.cu
    native/channel_native/kernels/bdpt_subpaths.cu
    native/channel_native/kernels/coupled_topology.cu
    native/channel_native/kernels/em_debug.cu
    native/channel_native/kernels/field_transport.cu
    native/channel_native/kernels/field_transport_ad.cu
    native/channel_native/kernels/diffraction.cu
    native/channel_native/kernels/diffraction_discovery.cu
    native/channel_native/kernels/deterministic_accum.cu
    native/channel_native/kernels/deterministic_field.cu
    native/channel_native/kernels/los.cu
    native/channel_native/kernels/material.cu
    native/channel_native/kernels/path_trace.cu
    native/channel_native/kernels/reflection.cu
    native/channel_native/kernels/scattering.cu
    native/channel_native/kernels/sampling.cu
    native/channel_native/montecarlo_basic.cpp
    native/channel_native/path_solver.cpp
)

target_compile_features(_channel_native PRIVATE cxx_std_17)
if(MSVC)
    # RayD's extern "C" integration entries raise std::runtime_error on
    # invalid inputs by contract. MSVC's default /EHc (part of /EHsc) assumes
    # extern "C" functions never raise, which turns those raises into
    # std::terminate at the bridge call sites; /EHc- keeps them catchable so
    # they surface as Python RuntimeError like every other validation error.
    set_source_files_properties(
        native/channel_native/raydn_bridge.cpp
        PROPERTIES COMPILE_OPTIONS "/EHc-")
endif()
target_link_libraries(
    _channel_native
    PRIVATE
        rayd_torch_native_core
        "${TORCH_LIBRARIES}"
        "${TORCH_PYTHON_LIBRARY}")
target_compile_definitions(_channel_native PRIVATE CHANNEL_NATIVE_WITH_RAYD=1 CHANNEL_NATIVE_WITH_OPTIX=1)

install(TARGETS _channel_native LIBRARY DESTINATION witwin/channel_native)
