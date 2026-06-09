cmake_minimum_required(VERSION 3.26)
project(raydtorch LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type for RayDTorch native targets." FORCE)
endif()

set(RAYDTORCH_DEFAULT_CUDA_ARCHITECTURES "75-real;80-real;86-real;89-real;120-real;120-virtual")
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES
   OR CMAKE_CUDA_ARCHITECTURES STREQUAL ""
   OR CMAKE_CUDA_ARCHITECTURES STREQUAL "52")
    set(
        CMAKE_CUDA_ARCHITECTURES
        "${RAYDTORCH_DEFAULT_CUDA_ARCHITECTURES}"
        CACHE STRING
        "CUDA architectures for RayDTorch native kernels."
        FORCE)
endif()

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

option(RAYDTORCH_BUILD_NATIVE "Build the RayDTorch extension." ON)
option(RAYDTORCH_OPTIX_FAST_MATH "Compile embedded OptiX PTX with --use_fast_math." ON)

set(RAYDTORCH_OPTIX_NVCC_FLAGS --std=c++17 --gpu-architecture=compute_75)
if(RAYDTORCH_OPTIX_FAST_MATH)
    list(APPEND RAYDTORCH_OPTIX_NVCC_FLAGS --use_fast_math)
endif()

if(RAYDTORCH_BUILD_NATIVE)
    find_package(CUDAToolkit 11.0 REQUIRED)
    find_path(
        OPTIX_INCLUDE_DIR
        optix.h
        HINTS
            ENV OPTIX_INCLUDE_DIR
            ENV OPTIX_PATH
            "$ENV{PROGRAMDATA}/NVIDIA Corporation/OptiX SDK 9.1.0"
            "$ENV{PROGRAMDATA}/NVIDIA Corporation/OptiX SDK 8.1.0"
            "$ENV{PROGRAMDATA}/NVIDIA Corporation/OptiX SDK 8.0.0"
            "/usr/local/NVIDIA-OptiX-SDK-9.1.0-linux64-x86_64"
            "/usr/local/NVIDIA-OptiX-SDK-8.1.0-linux64-x86_64"
            "/usr/local/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64"
            "/opt/NVIDIA-OptiX-SDK-9.1.0-linux64-x86_64"
            "/opt/NVIDIA-OptiX-SDK-8.1.0-linux64-x86_64"
            "/opt/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64"
        PATH_SUFFIXES include
    )
    if(NOT OPTIX_INCLUDE_DIR)
        message(FATAL_ERROR "Could not locate OptiX SDK include directory. Set OPTIX_INCLUDE_DIR or OPTIX_PATH.")
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

    function(pybind11_add_module target)
        Python_add_library(${target} MODULE WITH_SOABI ${ARGN})
        target_include_directories(
            ${target}
            PRIVATE
                "${TORCH_INSTALL_PREFIX}/include"
                "${TORCH_INSTALL_PREFIX}/include/torch/csrc/api/include"
        )
    endfunction()

    set(RAYDTORCH_OPTIX_PTX "${CMAKE_CURRENT_BINARY_DIR}/optix_intersect.ptx")
    set(RAYDTORCH_OPTIX_PTX_HEADER "${CMAKE_CURRENT_BINARY_DIR}/generated/raydtorch/optix_intersect_ptx.h")
    set(RAYDTORCH_EDGE_OPTIX_POINT_RAY_PTX "${CMAKE_CURRENT_BINARY_DIR}/edge_optix_point_ray.ptx")
    set(RAYDTORCH_EDGE_OPTIX_POINT_RAY_PTX_HEADER "${CMAKE_CURRENT_BINARY_DIR}/generated/raydtorch/edge_optix_point_ray_ptx.h")
    set(RAYDTORCH_EDGE_OPTIX_TOPK_PTX "${CMAKE_CURRENT_BINARY_DIR}/edge_optix_topk.ptx")
    set(RAYDTORCH_EDGE_OPTIX_TOPK_PTX_HEADER "${CMAKE_CURRENT_BINARY_DIR}/generated/raydtorch/edge_optix_topk_ptx.h")
    set(RAYDTORCH_REFLECTION_TRACE_PTX "${CMAKE_CURRENT_BINARY_DIR}/reflection_trace_optix.ptx")
    set(RAYDTORCH_REFLECTION_TRACE_PTX_HEADER "${CMAKE_CURRENT_BINARY_DIR}/generated/raydtorch/reflection_trace_optix_ptx.h")
    set(RAYDTORCH_SEGMENT_VISIBILITY_PTX "${CMAKE_CURRENT_BINARY_DIR}/segment_visibility_optix.ptx")
    set(RAYDTORCH_SEGMENT_VISIBILITY_PTX_HEADER "${CMAKE_CURRENT_BINARY_DIR}/generated/raydtorch/segment_visibility_optix_ptx.h")
    set(RAYDTORCH_REFLECTION_EPC_PTX "${CMAKE_CURRENT_BINARY_DIR}/reflection_epc_optix.ptx")
    set(RAYDTORCH_REFLECTION_EPC_PTX_HEADER "${CMAKE_CURRENT_BINARY_DIR}/generated/raydtorch/reflection_epc_optix_ptx.h")
    set(RAYDTORCH_REFLECTION_ACCUMULATION_PTX "${CMAKE_CURRENT_BINARY_DIR}/reflection_accumulation_optix.ptx")
    set(RAYDTORCH_REFLECTION_ACCUMULATION_PTX_HEADER "${CMAKE_CURRENT_BINARY_DIR}/generated/raydtorch/reflection_accumulation_optix_ptx.h")
    set(RAYDTORCH_DIFFRACTION_PATHS_PTX "${CMAKE_CURRENT_BINARY_DIR}/diffraction_paths_optix.ptx")
    set(RAYDTORCH_DIFFRACTION_PATHS_PTX_HEADER "${CMAKE_CURRENT_BINARY_DIR}/generated/raydtorch/diffraction_paths_optix_ptx.h")
    set(RAYDTORCH_DIFFRACTION_ACCUMULATION_PTX "${CMAKE_CURRENT_BINARY_DIR}/diffraction_accumulation_optix.ptx")
    set(RAYDTORCH_DIFFRACTION_ACCUMULATION_PTX_HEADER "${CMAKE_CURRENT_BINARY_DIR}/generated/raydtorch/diffraction_accumulation_optix_ptx.h")
    add_custom_command(
        OUTPUT "${RAYDTORCH_OPTIX_PTX}"
        COMMAND
            "${CMAKE_CUDA_COMPILER}"
            --ptx
            "${CMAKE_CURRENT_SOURCE_DIR}/src/torch_ext/scene/optix_intersect.cu"
            -o "${RAYDTORCH_OPTIX_PTX}"
            -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
            -I "${CUDAToolkit_INCLUDE_DIRS}"
            -I "${OPTIX_INCLUDE_DIR}"
            ${RAYDTORCH_OPTIX_NVCC_FLAGS}
        DEPENDS
            src/torch_ext/scene/optix_intersect.cu
            include/raydtorch/scene/optix_intersect_params.h
    )
    add_custom_command(
        OUTPUT "${RAYDTORCH_OPTIX_PTX_HEADER}"
        COMMAND
            "${Python_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/torch_ext/embed_ptx.py"
            "${RAYDTORCH_OPTIX_PTX}"
            "${RAYDTORCH_OPTIX_PTX_HEADER}"
            raydtorch_optix_intersect_ptx
        DEPENDS
            "${RAYDTORCH_OPTIX_PTX}"
            src/torch_ext/embed_ptx.py
    )
    add_custom_target(raydtorch_optix_ptx DEPENDS "${RAYDTORCH_OPTIX_PTX_HEADER}")
    add_custom_command(
        OUTPUT "${RAYDTORCH_EDGE_OPTIX_POINT_RAY_PTX}"
        COMMAND
            "${CMAKE_CUDA_COMPILER}"
            --ptx
            "${CMAKE_CURRENT_SOURCE_DIR}/src/torch_ext/edge/edge_optix.cu"
            -o "${RAYDTORCH_EDGE_OPTIX_POINT_RAY_PTX}"
            -DRAYDTORCH_EDGE_POINT_RAY_ONLY=1
            -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
            -I "${CUDAToolkit_INCLUDE_DIRS}"
            -I "${OPTIX_INCLUDE_DIR}"
            ${RAYDTORCH_OPTIX_NVCC_FLAGS}
        DEPENDS
            src/torch_ext/edge/edge_optix.cu
            include/raydtorch/common/math.cuh
            include/raydtorch/edge/optix_params.h
    )
    add_custom_command(
        OUTPUT "${RAYDTORCH_EDGE_OPTIX_POINT_RAY_PTX_HEADER}"
        COMMAND
            "${Python_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/torch_ext/embed_ptx.py"
            "${RAYDTORCH_EDGE_OPTIX_POINT_RAY_PTX}"
            "${RAYDTORCH_EDGE_OPTIX_POINT_RAY_PTX_HEADER}"
            raydtorch_edge_optix_point_ray_ptx
        DEPENDS
            "${RAYDTORCH_EDGE_OPTIX_POINT_RAY_PTX}"
            src/torch_ext/embed_ptx.py
    )
    add_custom_command(
        OUTPUT "${RAYDTORCH_EDGE_OPTIX_TOPK_PTX}"
        COMMAND
            "${CMAKE_CUDA_COMPILER}"
            --ptx
            "${CMAKE_CURRENT_SOURCE_DIR}/src/torch_ext/edge/edge_optix.cu"
            -o "${RAYDTORCH_EDGE_OPTIX_TOPK_PTX}"
            -DRAYDTORCH_EDGE_TOPK_ONLY=1
            -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
            -I "${CUDAToolkit_INCLUDE_DIRS}"
            -I "${OPTIX_INCLUDE_DIR}"
            ${RAYDTORCH_OPTIX_NVCC_FLAGS}
        DEPENDS
            src/torch_ext/edge/edge_optix.cu
            include/raydtorch/common/math.cuh
            include/raydtorch/edge/optix_params.h
    )
    add_custom_command(
        OUTPUT "${RAYDTORCH_EDGE_OPTIX_TOPK_PTX_HEADER}"
        COMMAND
            "${Python_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/torch_ext/embed_ptx.py"
            "${RAYDTORCH_EDGE_OPTIX_TOPK_PTX}"
            "${RAYDTORCH_EDGE_OPTIX_TOPK_PTX_HEADER}"
            raydtorch_edge_optix_topk_ptx
        DEPENDS
            "${RAYDTORCH_EDGE_OPTIX_TOPK_PTX}"
            src/torch_ext/embed_ptx.py
    )
    add_custom_target(
        raydtorch_edge_optix_ptx
        DEPENDS
            "${RAYDTORCH_EDGE_OPTIX_POINT_RAY_PTX_HEADER}"
            "${RAYDTORCH_EDGE_OPTIX_TOPK_PTX_HEADER}")
    add_custom_command(
        OUTPUT "${RAYDTORCH_REFLECTION_TRACE_PTX}"
        COMMAND
            "${CMAKE_CUDA_COMPILER}"
            --ptx
            "${CMAKE_CURRENT_SOURCE_DIR}/src/torch_ext/reflection/trace_optix.cu"
            -o "${RAYDTORCH_REFLECTION_TRACE_PTX}"
            -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
            -I "${CUDAToolkit_INCLUDE_DIRS}"
            -I "${OPTIX_INCLUDE_DIR}"
            ${RAYDTORCH_OPTIX_NVCC_FLAGS}
        DEPENDS
            src/torch_ext/reflection/trace_optix.cu
            include/raydtorch/common/math.cuh
            include/raydtorch/reflection/trace_params.h
    )
    add_custom_command(
        OUTPUT "${RAYDTORCH_REFLECTION_TRACE_PTX_HEADER}"
        COMMAND
            "${Python_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/torch_ext/embed_ptx.py"
            "${RAYDTORCH_REFLECTION_TRACE_PTX}"
            "${RAYDTORCH_REFLECTION_TRACE_PTX_HEADER}"
            raydtorch_reflection_trace_optix_ptx
        DEPENDS
            "${RAYDTORCH_REFLECTION_TRACE_PTX}"
            src/torch_ext/embed_ptx.py
    )
    add_custom_target(raydtorch_reflection_trace_optix_ptx DEPENDS "${RAYDTORCH_REFLECTION_TRACE_PTX_HEADER}")
    add_custom_command(
        OUTPUT "${RAYDTORCH_SEGMENT_VISIBILITY_PTX}"
        COMMAND
            "${CMAKE_CUDA_COMPILER}"
            --ptx
            "${CMAKE_CURRENT_SOURCE_DIR}/src/torch_ext/reflection/visibility_optix.cu"
            -o "${RAYDTORCH_SEGMENT_VISIBILITY_PTX}"
            -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
            -I "${CUDAToolkit_INCLUDE_DIRS}"
            -I "${OPTIX_INCLUDE_DIR}"
            ${RAYDTORCH_OPTIX_NVCC_FLAGS}
        DEPENDS
            src/torch_ext/reflection/visibility_optix.cu
            include/raydtorch/common/math.cuh
            include/raydtorch/reflection/visibility_params.h
    )
    add_custom_command(
        OUTPUT "${RAYDTORCH_SEGMENT_VISIBILITY_PTX_HEADER}"
        COMMAND
            "${Python_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/torch_ext/embed_ptx.py"
            "${RAYDTORCH_SEGMENT_VISIBILITY_PTX}"
            "${RAYDTORCH_SEGMENT_VISIBILITY_PTX_HEADER}"
            raydtorch_segment_visibility_optix_ptx
        DEPENDS
            "${RAYDTORCH_SEGMENT_VISIBILITY_PTX}"
            src/torch_ext/embed_ptx.py
    )
    add_custom_target(raydtorch_segment_visibility_optix_ptx DEPENDS "${RAYDTORCH_SEGMENT_VISIBILITY_PTX_HEADER}")
    add_custom_command(
        OUTPUT "${RAYDTORCH_REFLECTION_EPC_PTX}"
        COMMAND
            "${CMAKE_CUDA_COMPILER}"
            --ptx
            "${CMAKE_CURRENT_SOURCE_DIR}/src/torch_ext/reflection/epc_optix.cu"
            -o "${RAYDTORCH_REFLECTION_EPC_PTX}"
            -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
            -I "${CUDAToolkit_INCLUDE_DIRS}"
            -I "${OPTIX_INCLUDE_DIR}"
            ${RAYDTORCH_OPTIX_NVCC_FLAGS}
        DEPENDS
            src/torch_ext/reflection/epc_optix.cu
            include/raydtorch/common/math.cuh
            include/raydtorch/reflection/epc_params.h
    )
    add_custom_command(
        OUTPUT "${RAYDTORCH_REFLECTION_EPC_PTX_HEADER}"
        COMMAND
            "${Python_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/torch_ext/embed_ptx.py"
            "${RAYDTORCH_REFLECTION_EPC_PTX}"
            "${RAYDTORCH_REFLECTION_EPC_PTX_HEADER}"
            raydtorch_reflection_epc_optix_ptx
        DEPENDS
            "${RAYDTORCH_REFLECTION_EPC_PTX}"
            src/torch_ext/embed_ptx.py
    )
    add_custom_target(raydtorch_reflection_epc_optix_ptx DEPENDS "${RAYDTORCH_REFLECTION_EPC_PTX_HEADER}")
    add_custom_command(
        OUTPUT "${RAYDTORCH_REFLECTION_ACCUMULATION_PTX}"
        COMMAND
            "${CMAKE_CUDA_COMPILER}"
            --ptx
            "${CMAKE_CURRENT_SOURCE_DIR}/src/torch_ext/reflection/accum_optix.cu"
            -o "${RAYDTORCH_REFLECTION_ACCUMULATION_PTX}"
            -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
            -I "${CUDAToolkit_INCLUDE_DIRS}"
            -I "${OPTIX_INCLUDE_DIR}"
            ${RAYDTORCH_OPTIX_NVCC_FLAGS}
        DEPENDS
            src/torch_ext/reflection/accum_optix.cu
            include/raydtorch/common/complex.cuh
            include/raydtorch/common/math.cuh
            include/raydtorch/reflection/accum_params.h
    )
    add_custom_command(
        OUTPUT "${RAYDTORCH_REFLECTION_ACCUMULATION_PTX_HEADER}"
        COMMAND
            "${Python_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/torch_ext/embed_ptx.py"
            "${RAYDTORCH_REFLECTION_ACCUMULATION_PTX}"
            "${RAYDTORCH_REFLECTION_ACCUMULATION_PTX_HEADER}"
            raydtorch_reflection_accumulation_optix_ptx
        DEPENDS
            "${RAYDTORCH_REFLECTION_ACCUMULATION_PTX}"
            src/torch_ext/embed_ptx.py
    )
    add_custom_target(raydtorch_reflection_accumulation_optix_ptx DEPENDS "${RAYDTORCH_REFLECTION_ACCUMULATION_PTX_HEADER}")
    add_custom_command(
        OUTPUT "${RAYDTORCH_DIFFRACTION_PATHS_PTX}"
        COMMAND
            "${CMAKE_CUDA_COMPILER}"
            --ptx
            "${CMAKE_CURRENT_SOURCE_DIR}/src/torch_ext/diffraction/paths_optix.cu"
            -o "${RAYDTORCH_DIFFRACTION_PATHS_PTX}"
            -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
            -I "${CUDAToolkit_INCLUDE_DIRS}"
            -I "${OPTIX_INCLUDE_DIR}"
            ${RAYDTORCH_OPTIX_NVCC_FLAGS}
        DEPENDS
            src/torch_ext/diffraction/paths_optix.cu
            include/raydtorch/common/math.cuh
            include/raydtorch/diffraction/common.h
            include/raydtorch/diffraction/paths_params.h
    )
    add_custom_command(
        OUTPUT "${RAYDTORCH_DIFFRACTION_PATHS_PTX_HEADER}"
        COMMAND
            "${Python_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/torch_ext/embed_ptx.py"
            "${RAYDTORCH_DIFFRACTION_PATHS_PTX}"
            "${RAYDTORCH_DIFFRACTION_PATHS_PTX_HEADER}"
            raydtorch_diffraction_paths_optix_ptx
        DEPENDS
            "${RAYDTORCH_DIFFRACTION_PATHS_PTX}"
            src/torch_ext/embed_ptx.py
    )
    add_custom_target(raydtorch_diffraction_paths_optix_ptx DEPENDS "${RAYDTORCH_DIFFRACTION_PATHS_PTX_HEADER}")
    add_custom_command(
        OUTPUT "${RAYDTORCH_DIFFRACTION_ACCUMULATION_PTX}"
        COMMAND
            "${CMAKE_CUDA_COMPILER}"
            --ptx
            "${CMAKE_CURRENT_SOURCE_DIR}/src/torch_ext/diffraction/accum_optix.cu"
            -o "${RAYDTORCH_DIFFRACTION_ACCUMULATION_PTX}"
            -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
            -I "${CUDAToolkit_INCLUDE_DIRS}"
            -I "${OPTIX_INCLUDE_DIR}"
            ${RAYDTORCH_OPTIX_NVCC_FLAGS}
        DEPENDS
            src/torch_ext/diffraction/accum_optix.cu
            include/raydtorch/common/math.cuh
            include/raydtorch/diffraction/common.h
            include/raydtorch/diffraction/accum_params.h
            include/utd/utd_math.h
            include/utd/utd_types.h
    )
    add_custom_command(
        OUTPUT "${RAYDTORCH_DIFFRACTION_ACCUMULATION_PTX_HEADER}"
        COMMAND
            "${Python_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/torch_ext/embed_ptx.py"
            "${RAYDTORCH_DIFFRACTION_ACCUMULATION_PTX}"
            "${RAYDTORCH_DIFFRACTION_ACCUMULATION_PTX_HEADER}"
            raydtorch_diffraction_accumulation_optix_ptx
        DEPENDS
            "${RAYDTORCH_DIFFRACTION_ACCUMULATION_PTX}"
            src/torch_ext/embed_ptx.py
    )
    add_custom_target(raydtorch_diffraction_accumulation_optix_ptx DEPENDS "${RAYDTORCH_DIFFRACTION_ACCUMULATION_PTX_HEADER}")

    pybind11_add_module(
        _raydtorch
        src/torch_ext/module.cpp
        src/torch_ext/tensor_check.cpp
        src/torch_ext/scene/scene_cache.cpp
        src/torch_ext/common/optix_pipeline.cpp
        src/torch_ext/reflection/pipeline.cpp
        src/torch_ext/diffraction/pipeline.cpp
        src/torch_ext/scene/ops_scene.cpp
        src/torch_ext/scene/optix_context.cpp
        src/torch_ext/scene/ops_intersect.cpp
        src/torch_ext/edge/ops_edge.cpp
        src/torch_ext/reflection/ops.cpp
        src/torch_ext/diffraction/ops.cpp
        src/torch_ext/scene/cache_kernels.cu
        src/torch_ext/scene/geometry_forward.cu
        src/torch_ext/scene/geometry_backward.cu
        src/torch_ext/edge/bvh.cu
        src/torch_ext/edge/edge_forward.cu
        src/torch_ext/edge/edge_backward.cu
        src/torch_ext/reflection/visibility_backward.cu
        src/torch_ext/reflection/backward.cu
        src/torch_ext/reflection/dedup.cu
        src/torch_ext/reflection/epc_field.cu
        src/torch_ext/reflection/accum_reduce.cu
        src/torch_ext/diffraction/accum_reduce.cu
        src/torch_ext/diffraction/accum_ad.cu
    )
    target_include_directories(
        _raydtorch
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_BINARY_DIR}/generated
            ${CUDAToolkit_INCLUDE_DIRS}
            ${OPTIX_INCLUDE_DIR}
    )
    add_dependencies(
        _raydtorch
        raydtorch_optix_ptx
        raydtorch_edge_optix_ptx
        raydtorch_reflection_trace_optix_ptx
        raydtorch_segment_visibility_optix_ptx
        raydtorch_reflection_epc_optix_ptx
        raydtorch_reflection_accumulation_optix_ptx
        raydtorch_diffraction_paths_optix_ptx
        raydtorch_diffraction_accumulation_optix_ptx)
    target_link_libraries(_raydtorch PRIVATE "${TORCH_LIBRARIES}" "${TORCH_PYTHON_LIBRARY}" CUDA::cudart CUDA::cuda_driver)
    target_compile_features(_raydtorch PRIVATE cxx_std_17)
    install(TARGETS _raydtorch LIBRARY DESTINATION raydtorch RUNTIME DESTINATION raydtorch)
endif()
