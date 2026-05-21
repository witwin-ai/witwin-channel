cmake_minimum_required(VERSION 3.22)

if(NOT DEFINED SKBUILD_PROJECT_NAME)
    set(SKBUILD_PROJECT_NAME "witwin-channel")
endif()

if(NOT DEFINED SKBUILD_PROJECT_VERSION)
    set(SKBUILD_PROJECT_VERSION "0.1.0")
endif()

project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

function(resolve_python_package_path package_name suffix output_var)
    execute_process(
        COMMAND
            "${Python_EXECUTABLE}"
            -c
            "import importlib.util, pathlib, sys; spec = importlib.util.find_spec('${package_name}'); base = pathlib.Path(next(iter(spec.submodule_search_locations))).resolve() if spec and spec.submodule_search_locations else None; sys.stdout.write(str(base / '${suffix}')) if base else sys.exit(1)"
        RESULT_VARIABLE _result
        OUTPUT_VARIABLE _path
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    if(NOT _result EQUAL 0)
        message(FATAL_ERROR "Could not resolve ${package_name}/${suffix} from the active Python environment.")
    endif()

    set(${output_var} "${_path}" PARENT_SCOPE)
endfunction()

resolve_python_package_path("drjit" "cmake" DRJIT_CMAKE_DIR)
resolve_python_package_path("nanobind" "cmake" NANOBIND_CMAKE_DIR)

list(PREPEND CMAKE_PREFIX_PATH "${DRJIT_CMAKE_DIR}" "${NANOBIND_CMAKE_DIR}")

find_package(drjit CONFIG REQUIRED)
find_package(nanobind CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_subdirectory(witwin/channel/_native/channel_utils)
add_subdirectory(witwin/channel/_native/deterministic)
add_subdirectory(witwin/channel/_native/monte_carlo)
