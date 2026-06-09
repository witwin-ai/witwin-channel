cmake_minimum_required(VERSION 3.26)
project(raydn LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type for RayDN native targets." FORCE)
endif()

set(RAYDN_DEFAULT_CUDA_ARCHITECTURES "75-real;80-real;86-real;89-real;120-real;120-virtual")
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES
   OR CMAKE_CUDA_ARCHITECTURES STREQUAL ""
   OR CMAKE_CUDA_ARCHITECTURES STREQUAL "52")
    set(
        CMAKE_CUDA_ARCHITECTURES
        "${RAYDN_DEFAULT_CUDA_ARCHITECTURES}"
        CACHE STRING
        "CUDA architectures for RayDN native kernels."
        FORCE)
endif()

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

option(RAYDN_BUILD_NATIVE "Build the RayDN extension." ON)
option(RAYDN_OPTIX_FAST_MATH "Compile embedded OptiX PTX with --use_fast_math." ON)

set(RAYDN_OPTIX_NVCC_FLAGS --std=c++17 --gpu-architecture=compute_75)
if(RAYDN_OPTIX_FAST_MATH)
    list(APPEND RAYDN_OPTIX_NVCC_FLAGS --use_fast_math)
endif()

if(RAYDN_BUILD_NATIVE)
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

    set(RAYDN_OPTIX_PTX "${CMAKE_CURRENT_BINARY_DIR}/optix_intersect.ptx")
    set(RAYDN_OPTIX_PTX_HEADER "${CMAKE_CURRENT_BINARY_DIR}/generated/raydn/optix_intersect_ptx.h")
    set(RAYDN_EDGE_OPTIX_POINT_RAY_PTX "${CMAKE_CURRENT_BINARY_DIR}/edge_optix_point_ray.ptx")
    set(RAYDN_EDGE_OPTIX_POINT_RAY_PTX_HEADER "${CMAKE_CURRENT_BINARY_DIR}/generated/raydn/edge_optix_point_ray_ptx.h")
    set(RAYDN_EDGE_OPTIX_TOPK_PTX "${CMAKE_CURRENT_BINARY_DIR}/edge_optix_topk.ptx")
    set(RAYDN_EDGE_OPTIX_TOPK_PTX_HEADER "${CMAKE_CURRENT_BINARY_DIR}/generated/raydn/edge_optix_topk_ptx.h")
    set(RAYDN_REFLECTION_TRACE_PTX "${CMAKE_CURRENT_BINARY_DIR}/reflection_trace_optix.ptx")
    set(RAYDN_REFLECTION_TRACE_PTX_HEADER "${CMAKE_CURRENT_BINARY_DIR}/generated/raydn/reflection_trace_optix_ptx.h")
    set(RAYDN_SEGMENT_VISIBILITY_PTX "${CMAKE_CURRENT_BINARY_DIR}/segment_visibility_optix.ptx")
    set(RAYDN_SEGMENT_VISIBILITY_PTX_HEADER "${CMAKE_CURRENT_BINARY_DIR}/generated/raydn/segment_visibility_optix_ptx.h")
    set(RAYDN_REFLECTION_EPC_PTX "${CMAKE_CURRENT_BINARY_DIR}/reflection_epc_optix.ptx")
    set(RAYDN_REFLECTION_EPC_PTX_HEADER "${CMAKE_CURRENT_BINARY_DIR}/generated/raydn/reflection_epc_optix_ptx.h")
    set(RAYDN_REFLECTION_ACCUMULATION_PTX "${CMAKE_CURRENT_BINARY_DIR}/reflection_accumulation_optix.ptx")
    set(RAYDN_REFLECTION_ACCUMULATION_PTX_HEADER "${CMAKE_CURRENT_BINARY_DIR}/generated/raydn/reflection_accumulation_optix_ptx.h")
    set(RAYDN_DIFFRACTION_PATHS_PTX "${CMAKE_CURRENT_BINARY_DIR}/diffraction_paths_optix.ptx")
    set(RAYDN_DIFFRACTION_PATHS_PTX_HEADER "${CMAKE_CURRENT_BINARY_DIR}/generated/raydn/diffraction_paths_optix_ptx.h")
    set(RAYDN_DIFFRACTION_ACCUMULATION_PTX "${CMAKE_CURRENT_BINARY_DIR}/diffraction_accumulation_optix.ptx")
    set(RAYDN_DIFFRACTION_ACCUMULATION_PTX_HEADER "${CMAKE_CURRENT_BINARY_DIR}/generated/raydn/diffraction_accumulation_optix_ptx.h")
    add_custom_command(
        OUTPUT "${RAYDN_OPTIX_PTX}"
        COMMAND
            "${CMAKE_CUDA_COMPILER}"
            --ptx
            "${CMAKE_CURRENT_SOURCE_DIR}/src/torch_ext/scene/optix_intersect.cu"
            -o "${RAYDN_OPTIX_PTX}"
            -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
            -I "${CUDAToolkit_INCLUDE_DIRS}"
            -I "${OPTIX_INCLUDE_DIR}"
            ${RAYDN_OPTIX_NVCC_FLAGS}
        DEPENDS
            src/torch_ext/scene/optix_intersect.cu
            include/raydn/scene/optix_intersect_params.h
    )
    add_custom_command(
        OUTPUT "${RAYDN_OPTIX_PTX_HEADER}"
        COMMAND
            "${Python_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/torch_ext/embed_ptx.py"
            "${RAYDN_OPTIX_PTX}"
            "${RAYDN_OPTIX_PTX_HEADER}"
            raydn_optix_intersect_ptx
        DEPENDS
            "${RAYDN_OPTIX_PTX}"
            src/torch_ext/embed_ptx.py
    )
    add_custom_target(raydn_optix_ptx DEPENDS "${RAYDN_OPTIX_PTX_HEADER}")
    add_custom_command(
        OUTPUT "${RAYDN_EDGE_OPTIX_POINT_RAY_PTX}"
        COMMAND
            "${CMAKE_CUDA_COMPILER}"
            --ptx
            "${CMAKE_CURRENT_SOURCE_DIR}/src/torch_ext/edge/edge_optix.cu"
            -o "${RAYDN_EDGE_OPTIX_POINT_RAY_PTX}"
            -DRAYDN_EDGE_POINT_RAY_ONLY=1
            -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
            -I "${CUDAToolkit_INCLUDE_DIRS}"
            -I "${OPTIX_INCLUDE_DIR}"
            ${RAYDN_OPTIX_NVCC_FLAGS}
        DEPENDS
            src/torch_ext/edge/edge_optix.cu
            include/raydn/common/math.cuh
            include/raydn/edge/optix_params.h
    )
    add_custom_command(
        OUTPUT "${RAYDN_EDGE_OPTIX_POINT_RAY_PTX_HEADER}"
        COMMAND
            "${Python_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/torch_ext/embed_ptx.py"
            "${RAYDN_EDGE_OPTIX_POINT_RAY_PTX}"
            "${RAYDN_EDGE_OPTIX_POINT_RAY_PTX_HEADER}"
            raydn_edge_optix_point_ray_ptx
        DEPENDS
            "${RAYDN_EDGE_OPTIX_POINT_RAY_PTX}"
            src/torch_ext/embed_ptx.py
    )
    add_custom_command(
        OUTPUT "${RAYDN_EDGE_OPTIX_TOPK_PTX}"
        COMMAND
            "${CMAKE_CUDA_COMPILER}"
            --ptx
            "${CMAKE_CURRENT_SOURCE_DIR}/src/torch_ext/edge/edge_optix.cu"
            -o "${RAYDN_EDGE_OPTIX_TOPK_PTX}"
            -DRAYDN_EDGE_TOPK_ONLY=1
            -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
            -I "${CUDAToolkit_INCLUDE_DIRS}"
            -I "${OPTIX_INCLUDE_DIR}"
            ${RAYDN_OPTIX_NVCC_FLAGS}
        DEPENDS
            src/torch_ext/edge/edge_optix.cu
            include/raydn/common/math.cuh
            include/raydn/edge/optix_params.h
    )
    add_custom_command(
        OUTPUT "${RAYDN_EDGE_OPTIX_TOPK_PTX_HEADER}"
        COMMAND
            "${Python_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/torch_ext/embed_ptx.py"
            "${RAYDN_EDGE_OPTIX_TOPK_PTX}"
            "${RAYDN_EDGE_OPTIX_TOPK_PTX_HEADER}"
            raydn_edge_optix_topk_ptx
        DEPENDS
            "${RAYDN_EDGE_OPTIX_TOPK_PTX}"
            src/torch_ext/embed_ptx.py
    )
    add_custom_target(
        raydn_edge_optix_ptx
        DEPENDS
            "${RAYDN_EDGE_OPTIX_POINT_RAY_PTX_HEADER}"
            "${RAYDN_EDGE_OPTIX_TOPK_PTX_HEADER}")
    add_custom_command(
        OUTPUT "${RAYDN_REFLECTION_TRACE_PTX}"
        COMMAND
            "${CMAKE_CUDA_COMPILER}"
            --ptx
            "${CMAKE_CURRENT_SOURCE_DIR}/src/torch_ext/reflection/trace_optix.cu"
            -o "${RAYDN_REFLECTION_TRACE_PTX}"
            -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
            -I "${CUDAToolkit_INCLUDE_DIRS}"
            -I "${OPTIX_INCLUDE_DIR}"
            ${RAYDN_OPTIX_NVCC_FLAGS}
        DEPENDS
            src/torch_ext/reflection/trace_optix.cu
            include/raydn/common/math.cuh
            include/raydn/reflection/trace_params.h
    )
    add_custom_command(
        OUTPUT "${RAYDN_REFLECTION_TRACE_PTX_HEADER}"
        COMMAND
            "${Python_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/torch_ext/embed_ptx.py"
            "${RAYDN_REFLECTION_TRACE_PTX}"
            "${RAYDN_REFLECTION_TRACE_PTX_HEADER}"
            raydn_reflection_trace_optix_ptx
        DEPENDS
            "${RAYDN_REFLECTION_TRACE_PTX}"
            src/torch_ext/embed_ptx.py
    )
    add_custom_target(raydn_reflection_trace_optix_ptx DEPENDS "${RAYDN_REFLECTION_TRACE_PTX_HEADER}")
    add_custom_command(
        OUTPUT "${RAYDN_SEGMENT_VISIBILITY_PTX}"
        COMMAND
            "${CMAKE_CUDA_COMPILER}"
            --ptx
            "${CMAKE_CURRENT_SOURCE_DIR}/src/torch_ext/reflection/visibility_optix.cu"
            -o "${RAYDN_SEGMENT_VISIBILITY_PTX}"
            -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
            -I "${CUDAToolkit_INCLUDE_DIRS}"
            -I "${OPTIX_INCLUDE_DIR}"
            ${RAYDN_OPTIX_NVCC_FLAGS}
        DEPENDS
            src/torch_ext/reflection/visibility_optix.cu
            include/raydn/common/math.cuh
            include/raydn/reflection/visibility_params.h
    )
    add_custom_command(
        OUTPUT "${RAYDN_SEGMENT_VISIBILITY_PTX_HEADER}"
        COMMAND
            "${Python_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/torch_ext/embed_ptx.py"
            "${RAYDN_SEGMENT_VISIBILITY_PTX}"
            "${RAYDN_SEGMENT_VISIBILITY_PTX_HEADER}"
            raydn_segment_visibility_optix_ptx
        DEPENDS
            "${RAYDN_SEGMENT_VISIBILITY_PTX}"
            src/torch_ext/embed_ptx.py
    )
    add_custom_target(raydn_segment_visibility_optix_ptx DEPENDS "${RAYDN_SEGMENT_VISIBILITY_PTX_HEADER}")
    add_custom_command(
        OUTPUT "${RAYDN_REFLECTION_EPC_PTX}"
        COMMAND
            "${CMAKE_CUDA_COMPILER}"
            --ptx
            "${CMAKE_CURRENT_SOURCE_DIR}/src/torch_ext/reflection/epc_optix.cu"
            -o "${RAYDN_REFLECTION_EPC_PTX}"
            -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
            -I "${CUDAToolkit_INCLUDE_DIRS}"
            -I "${OPTIX_INCLUDE_DIR}"
            ${RAYDN_OPTIX_NVCC_FLAGS}
        DEPENDS
            src/torch_ext/reflection/epc_optix.cu
            include/raydn/common/math.cuh
            include/raydn/reflection/epc_params.h
    )
    add_custom_command(
        OUTPUT "${RAYDN_REFLECTION_EPC_PTX_HEADER}"
        COMMAND
            "${Python_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/torch_ext/embed_ptx.py"
            "${RAYDN_REFLECTION_EPC_PTX}"
            "${RAYDN_REFLECTION_EPC_PTX_HEADER}"
            raydn_reflection_epc_optix_ptx
        DEPENDS
            "${RAYDN_REFLECTION_EPC_PTX}"
            src/torch_ext/embed_ptx.py
    )
    add_custom_target(raydn_reflection_epc_optix_ptx DEPENDS "${RAYDN_REFLECTION_EPC_PTX_HEADER}")
    add_custom_command(
        OUTPUT "${RAYDN_REFLECTION_ACCUMULATION_PTX}"
        COMMAND
            "${CMAKE_CUDA_COMPILER}"
            --ptx
            "${CMAKE_CURRENT_SOURCE_DIR}/src/torch_ext/reflection/accum_optix.cu"
            -o "${RAYDN_REFLECTION_ACCUMULATION_PTX}"
            -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
            -I "${CUDAToolkit_INCLUDE_DIRS}"
            -I "${OPTIX_INCLUDE_DIR}"
            ${RAYDN_OPTIX_NVCC_FLAGS}
        DEPENDS
            src/torch_ext/reflection/accum_optix.cu
            include/raydn/common/complex.cuh
            include/raydn/common/math.cuh
            include/raydn/reflection/accum_params.h
    )
    add_custom_command(
        OUTPUT "${RAYDN_REFLECTION_ACCUMULATION_PTX_HEADER}"
        COMMAND
            "${Python_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/torch_ext/embed_ptx.py"
            "${RAYDN_REFLECTION_ACCUMULATION_PTX}"
            "${RAYDN_REFLECTION_ACCUMULATION_PTX_HEADER}"
            raydn_reflection_accumulation_optix_ptx
        DEPENDS
            "${RAYDN_REFLECTION_ACCUMULATION_PTX}"
            src/torch_ext/embed_ptx.py
    )
    add_custom_target(raydn_reflection_accumulation_optix_ptx DEPENDS "${RAYDN_REFLECTION_ACCUMULATION_PTX_HEADER}")
    add_custom_command(
        OUTPUT "${RAYDN_DIFFRACTION_PATHS_PTX}"
        COMMAND
            "${CMAKE_CUDA_COMPILER}"
            --ptx
            "${CMAKE_CURRENT_SOURCE_DIR}/src/torch_ext/diffraction/paths_optix.cu"
            -o "${RAYDN_DIFFRACTION_PATHS_PTX}"
            -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
            -I "${CUDAToolkit_INCLUDE_DIRS}"
            -I "${OPTIX_INCLUDE_DIR}"
            ${RAYDN_OPTIX_NVCC_FLAGS}
        DEPENDS
            src/torch_ext/diffraction/paths_optix.cu
            include/raydn/common/math.cuh
            include/raydn/diffraction/common.h
            include/raydn/diffraction/paths_params.h
    )
    add_custom_command(
        OUTPUT "${RAYDN_DIFFRACTION_PATHS_PTX_HEADER}"
        COMMAND
            "${Python_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/torch_ext/embed_ptx.py"
            "${RAYDN_DIFFRACTION_PATHS_PTX}"
            "${RAYDN_DIFFRACTION_PATHS_PTX_HEADER}"
            raydn_diffraction_paths_optix_ptx
        DEPENDS
            "${RAYDN_DIFFRACTION_PATHS_PTX}"
            src/torch_ext/embed_ptx.py
    )
    add_custom_target(raydn_diffraction_paths_optix_ptx DEPENDS "${RAYDN_DIFFRACTION_PATHS_PTX_HEADER}")
    add_custom_command(
        OUTPUT "${RAYDN_DIFFRACTION_ACCUMULATION_PTX}"
        COMMAND
            "${CMAKE_CUDA_COMPILER}"
            --ptx
            "${CMAKE_CURRENT_SOURCE_DIR}/src/torch_ext/diffraction/accum_optix.cu"
            -o "${RAYDN_DIFFRACTION_ACCUMULATION_PTX}"
            -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
            -I "${CUDAToolkit_INCLUDE_DIRS}"
            -I "${OPTIX_INCLUDE_DIR}"
            ${RAYDN_OPTIX_NVCC_FLAGS}
        DEPENDS
            src/torch_ext/diffraction/accum_optix.cu
            include/raydn/common/math.cuh
            include/raydn/diffraction/common.h
            include/raydn/diffraction/accum_params.h
            include/utd/utd_math.h
            include/utd/utd_types.h
    )
    add_custom_command(
        OUTPUT "${RAYDN_DIFFRACTION_ACCUMULATION_PTX_HEADER}"
        COMMAND
            "${Python_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/torch_ext/embed_ptx.py"
            "${RAYDN_DIFFRACTION_ACCUMULATION_PTX}"
            "${RAYDN_DIFFRACTION_ACCUMULATION_PTX_HEADER}"
            raydn_diffraction_accumulation_optix_ptx
        DEPENDS
            "${RAYDN_DIFFRACTION_ACCUMULATION_PTX}"
            src/torch_ext/embed_ptx.py
    )
    add_custom_target(raydn_diffraction_accumulation_optix_ptx DEPENDS "${RAYDN_DIFFRACTION_ACCUMULATION_PTX_HEADER}")

    pybind11_add_module(
        _raydn
        src/torch_ext/module.cpp
        src/torch_ext/library.cpp
        src/torch_ext/tensor_check.cpp
        src/torch_ext/scene/scene_cache.cpp
        src/torch_ext/common/optix_pipeline.cpp
        src/torch_ext/common/ops_stats.cpp
        src/torch_ext/common/ops_camera.cpp
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
        src/torch_ext/diffraction/paths_init.cu
        src/torch_ext/common/stats.cu
        src/torch_ext/common/camera.cu
    )
    target_include_directories(
        _raydn
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_BINARY_DIR}/generated
            ${CUDAToolkit_INCLUDE_DIRS}
            ${OPTIX_INCLUDE_DIR}
    )
    add_dependencies(
        _raydn
        raydn_optix_ptx
        raydn_edge_optix_ptx
        raydn_reflection_trace_optix_ptx
        raydn_segment_visibility_optix_ptx
        raydn_reflection_epc_optix_ptx
        raydn_reflection_accumulation_optix_ptx
        raydn_diffraction_paths_optix_ptx
        raydn_diffraction_accumulation_optix_ptx)
    target_link_libraries(_raydn PRIVATE "${TORCH_LIBRARIES}" "${TORCH_PYTHON_LIBRARY}" CUDA::cudart CUDA::cuda_driver)
    target_compile_features(_raydn PRIVATE cxx_std_17)
    install(TARGETS _raydn LIBRARY DESTINATION raydn RUNTIME DESTINATION raydn)
endif()
