cmake_minimum_required(VERSION 3.26)

set(CHANNEL_NATIVE_ABI_VERSION 1)
set(CHANNEL_NATIVE_RAYD_LOCK_FILE "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/rayd.lock.json")
if(NOT EXISTS "${CHANNEL_NATIVE_RAYD_LOCK_FILE}")
    message(FATAL_ERROR "Missing RayD dependency lock: ${CHANNEL_NATIVE_RAYD_LOCK_FILE}")
endif()
file(READ "${CHANNEL_NATIVE_RAYD_LOCK_FILE}" CHANNEL_NATIVE_RAYD_LOCK_JSON)
file(SHA256 "${CHANNEL_NATIVE_RAYD_LOCK_FILE}" CHANNEL_NATIVE_RAYD_LOCK_SHA256)
string(JSON RAYD_LOCK_SCHEMA_VERSION GET "${CHANNEL_NATIVE_RAYD_LOCK_JSON}" schema_version)
string(JSON RAYD_LOCK_REPOSITORY_URL GET "${CHANNEL_NATIVE_RAYD_LOCK_JSON}" repository_url)
string(JSON RAYD_LOCK_COMMIT GET "${CHANNEL_NATIVE_RAYD_LOCK_JSON}" commit)
string(JSON RAYD_LOCK_ABI_KIND GET "${CHANNEL_NATIVE_RAYD_LOCK_JSON}" integration_abi kind)
string(JSON RAYD_LOCK_ABI_PATH GET "${CHANNEL_NATIVE_RAYD_LOCK_JSON}" integration_abi path)
string(JSON RAYD_LOCK_ABI_SHA256 GET "${CHANNEL_NATIVE_RAYD_LOCK_JSON}" integration_abi sha256)
if(NOT RAYD_LOCK_SCHEMA_VERSION EQUAL 1)
    message(FATAL_ERROR "Unsupported RayD lock schema_version=${RAYD_LOCK_SCHEMA_VERSION}; expected 1.")
endif()
if(NOT RAYD_LOCK_ABI_KIND STREQUAL "source-header-sha256")
    message(FATAL_ERROR
        "Unsupported RayD integration ABI kind '${RAYD_LOCK_ABI_KIND}'; "
        "expected source-header-sha256.")
endif()

set(CHANNEL_NATIVE_DEFAULT_CUDA_ARCHITECTURES "75-real;80-real;86-real;89-real;120-real;120-virtual")
set(CHANNEL_NATIVE_CMAKE_CUDA_ARCHITECTURES_EXPLICIT FALSE)
if(DEFINED CMAKE_CUDA_ARCHITECTURES
   AND NOT CMAKE_CUDA_ARCHITECTURES STREQUAL ""
   AND NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "52")
    set(CHANNEL_NATIVE_CMAKE_CUDA_ARCHITECTURES_EXPLICIT TRUE)
endif()
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "" OR CMAKE_CUDA_ARCHITECTURES STREQUAL "52")
    set(CMAKE_CUDA_ARCHITECTURES "${CHANNEL_NATIVE_DEFAULT_CUDA_ARCHITECTURES}" CACHE STRING "CUDA architectures for Channel Native CUDA targets." FORCE)
endif()
set(CHANNEL_NATIVE_DEFAULT_TORCH_CUDA_ARCH_LIST "7.5 8.0 8.6 8.9 12.0+PTX")

function(channel_native_normalize_cmake_architectures raw_architectures output_variable)
    set(normalized_architectures "")
    set(architecture_tokens "${raw_architectures}")
    foreach(architecture_token IN LISTS architecture_tokens)
        if(NOT architecture_token MATCHES "^([0-9]+)(-(real|virtual))?$")
            message(FATAL_ERROR
                "Unsupported CMAKE_CUDA_ARCHITECTURES entry '${architecture_token}'. "
                "Use explicit numeric entries such as '86-real' or '120-virtual'.")
        endif()
        set(architecture_number "${CMAKE_MATCH_1}")
        set(architecture_kind "${CMAKE_MATCH_3}")
        if(architecture_kind STREQUAL "")
            list(APPEND normalized_architectures
                "${architecture_number}-real"
                "${architecture_number}-virtual")
        else()
            list(APPEND normalized_architectures
                "${architecture_number}-${architecture_kind}")
        endif()
    endforeach()
    list(REMOVE_DUPLICATES normalized_architectures)
    if(NOT normalized_architectures)
        message(FATAL_ERROR "CMAKE_CUDA_ARCHITECTURES produced an empty architecture identity.")
    endif()
    set(${output_variable} "${normalized_architectures}" PARENT_SCOPE)
endfunction()

function(channel_native_torch_arch_list normalized_architectures output_variable)
    set(torch_architectures "")
    foreach(architecture IN LISTS normalized_architectures)
        if(architecture MATCHES "^([0-9]+)-virtual$")
            set(architecture_number "${CMAKE_MATCH_1}")
            list(FIND normalized_architectures "${architecture_number}-real" real_index)
            if(real_index EQUAL -1)
                message(FATAL_ERROR
                    "Channel Native does not support PTX-only architecture '${architecture}'.")
            endif()
        endif()
    endforeach()
    foreach(architecture IN LISTS normalized_architectures)
        if(NOT architecture MATCHES "^([0-9]+)-real$")
            continue()
        endif()
        set(architecture_number "${CMAKE_MATCH_1}")
        string(LENGTH "${architecture_number}" architecture_length)
        math(EXPR minor_index "${architecture_length} - 1")
        string(SUBSTRING "${architecture_number}" 0 ${minor_index} architecture_major)
        string(SUBSTRING "${architecture_number}" ${minor_index} 1 architecture_minor)
        set(torch_architecture "${architecture_major}.${architecture_minor}")
        list(FIND normalized_architectures "${architecture_number}-virtual" virtual_index)
        if(NOT virtual_index EQUAL -1)
            string(APPEND torch_architecture "+PTX")
        endif()
        list(APPEND torch_architectures "${torch_architecture}")
    endforeach()
    string(JOIN " " torch_arch_list ${torch_architectures})
    set(${output_variable} "${torch_arch_list}" PARENT_SCOPE)
endfunction()

if("$ENV{TORCH_CUDA_ARCH_LIST}" STREQUAL "")
    if(CHANNEL_NATIVE_CMAKE_CUDA_ARCHITECTURES_EXPLICIT)
        channel_native_normalize_cmake_architectures(
            "${CMAKE_CUDA_ARCHITECTURES}"
            CHANNEL_NATIVE_CMAKE_CUDA_ARCHITECTURES)
        channel_native_torch_arch_list(
            "${CHANNEL_NATIVE_CMAKE_CUDA_ARCHITECTURES}"
            CHANNEL_NATIVE_DERIVED_TORCH_CUDA_ARCH_LIST)
        set(ENV{TORCH_CUDA_ARCH_LIST} "${CHANNEL_NATIVE_DERIVED_TORCH_CUDA_ARCH_LIST}")
    else()
        set(ENV{TORCH_CUDA_ARCH_LIST} "${CHANNEL_NATIVE_DEFAULT_TORCH_CUDA_ARCH_LIST}")
    endif()
endif()
set(CHANNEL_NATIVE_TORCH_CUDA_ARCH_LIST "$ENV{TORCH_CUDA_ARCH_LIST}")
if(CHANNEL_NATIVE_TORCH_CUDA_ARCH_LIST STREQUAL "OFF")
    message(FATAL_ERROR "TORCH_CUDA_ARCH_LIST must not be OFF for Channel Native.")
endif()
separate_arguments(
    CHANNEL_NATIVE_TORCH_CUDA_ARCH_TOKENS
    NATIVE_COMMAND
    "${CHANNEL_NATIVE_TORCH_CUDA_ARCH_LIST}")
set(CHANNEL_NATIVE_CUDA_ARCHITECTURES "")
foreach(CHANNEL_NATIVE_CUDA_ARCH_TOKEN IN LISTS CHANNEL_NATIVE_TORCH_CUDA_ARCH_TOKENS)
    if(CHANNEL_NATIVE_CUDA_ARCH_TOKEN STREQUAL "")
        continue()
    endif()
    if(NOT CHANNEL_NATIVE_CUDA_ARCH_TOKEN MATCHES "^([0-9]+)\\.([0-9]+)(\\+PTX)?$")
        message(FATAL_ERROR
            "Unsupported TORCH_CUDA_ARCH_LIST entry '${CHANNEL_NATIVE_CUDA_ARCH_TOKEN}'. "
            "Use numeric entries such as '8.6' or '12.0+PTX'.")
    endif()
    set(CHANNEL_NATIVE_CUDA_ARCH_NUMBER "${CMAKE_MATCH_1}${CMAKE_MATCH_2}")
    set(CHANNEL_NATIVE_CUDA_ARCH_HAS_PTX "${CMAKE_MATCH_3}")
    list(APPEND CHANNEL_NATIVE_CUDA_ARCHITECTURES "${CHANNEL_NATIVE_CUDA_ARCH_NUMBER}-real")
    if(NOT "${CHANNEL_NATIVE_CUDA_ARCH_HAS_PTX}" STREQUAL "")
        list(APPEND CHANNEL_NATIVE_CUDA_ARCHITECTURES "${CHANNEL_NATIVE_CUDA_ARCH_NUMBER}-virtual")
    endif()
endforeach()
list(REMOVE_DUPLICATES CHANNEL_NATIVE_CUDA_ARCHITECTURES)
if(NOT CHANNEL_NATIVE_CUDA_ARCHITECTURES)
    message(FATAL_ERROR "TORCH_CUDA_ARCH_LIST produced an empty Channel Native architecture identity.")
endif()
channel_native_normalize_cmake_architectures(
    "${CMAKE_CUDA_ARCHITECTURES}"
    CHANNEL_NATIVE_CMAKE_CUDA_ARCHITECTURES)
if(CHANNEL_NATIVE_CMAKE_CUDA_ARCHITECTURES_EXPLICIT)
    set(CHANNEL_NATIVE_CMAKE_CUDA_ARCHITECTURES_SORTED
        ${CHANNEL_NATIVE_CMAKE_CUDA_ARCHITECTURES})
    set(CHANNEL_NATIVE_TORCH_CUDA_ARCHITECTURES_SORTED
        ${CHANNEL_NATIVE_CUDA_ARCHITECTURES})
    list(SORT CHANNEL_NATIVE_CMAKE_CUDA_ARCHITECTURES_SORTED)
    list(SORT CHANNEL_NATIVE_TORCH_CUDA_ARCHITECTURES_SORTED)
    if(NOT "${CHANNEL_NATIVE_CMAKE_CUDA_ARCHITECTURES_SORTED}"
       STREQUAL "${CHANNEL_NATIVE_TORCH_CUDA_ARCHITECTURES_SORTED}")
        message(FATAL_ERROR
            "CMAKE_CUDA_ARCHITECTURES and TORCH_CUDA_ARCH_LIST disagree: "
            "'${CHANNEL_NATIVE_CMAKE_CUDA_ARCHITECTURES}' versus "
            "'${CHANNEL_NATIVE_CUDA_ARCHITECTURES}'.")
    endif()
else()
    set(
        CMAKE_CUDA_ARCHITECTURES
        "${CHANNEL_NATIVE_CUDA_ARCHITECTURES}"
        CACHE STRING
        "CUDA architectures for Channel Native and RayD CUDA targets."
        FORCE)
endif()
channel_native_torch_arch_list(
    "${CHANNEL_NATIVE_CUDA_ARCHITECTURES}"
    CHANNEL_NATIVE_TORCH_CUDA_ARCH_LIST)
set(TORCH_CUDA_ARCH_LIST "${CHANNEL_NATIVE_TORCH_CUDA_ARCH_LIST}")
set(ENV{TORCH_CUDA_ARCH_LIST} "${CHANNEL_NATIVE_TORCH_CUDA_ARCH_LIST}")
message(STATUS "Channel Native CUDA architectures: ${CHANNEL_NATIVE_CUDA_ARCHITECTURES}")
message(STATUS "Channel Native Torch CUDA architectures: ${CHANNEL_NATIVE_TORCH_CUDA_ARCH_LIST}")

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(Git REQUIRED)
if(WIN32)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c
            "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '')"
        RESULT_VARIABLE CHANNEL_NATIVE_PYTHON_SUFFIX_RESULT
        OUTPUT_VARIABLE CHANNEL_NATIVE_PYTHON_EXTENSION_SUFFIX
        ERROR_VARIABLE CHANNEL_NATIVE_PYTHON_SUFFIX_ERROR
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT CHANNEL_NATIVE_PYTHON_SUFFIX_RESULT EQUAL 0
       OR NOT CHANNEL_NATIVE_PYTHON_EXTENSION_SUFFIX MATCHES "^\\..*\\.pyd$")
        message(FATAL_ERROR
            "Could not determine the configured Python extension suffix: "
            "'${CHANNEL_NATIVE_PYTHON_EXTENSION_SUFFIX}' ${CHANNEL_NATIVE_PYTHON_SUFFIX_ERROR}")
    endif()
    get_filename_component(CHANNEL_NATIVE_PYTHON_ENV_ROOT "${Python_EXECUTABLE}" DIRECTORY)
    set(CHANNEL_NATIVE_PYTHON_LIBRARY_DIR "${CHANNEL_NATIVE_PYTHON_ENV_ROOT}/libs")
    find_library(
        CHANNEL_NATIVE_PYTHON_LIBRARY
        NAMES "python${Python_VERSION_MAJOR}${Python_VERSION_MINOR}"
        PATHS "${CHANNEL_NATIVE_PYTHON_LIBRARY_DIR}"
        NO_DEFAULT_PATH
        REQUIRED)
endif()

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
execute_process(
    COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
    WORKING_DIRECTORY "${RAYD_SOURCE_DIR}"
    RESULT_VARIABLE RAYD_GIT_HEAD_RESULT
    OUTPUT_VARIABLE RAYD_GIT_HEAD
    ERROR_VARIABLE RAYD_GIT_HEAD_ERROR
    OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT RAYD_GIT_HEAD_RESULT EQUAL 0)
    message(FATAL_ERROR
        "Could not read the RayD Git HEAD in '${RAYD_SOURCE_DIR}': ${RAYD_GIT_HEAD_ERROR}")
endif()
if(NOT RAYD_GIT_HEAD STREQUAL RAYD_LOCK_COMMIT)
    message(FATAL_ERROR
        "RayD revision mismatch: checkout is ${RAYD_GIT_HEAD}, "
        "lock requires ${RAYD_LOCK_COMMIT}.")
endif()
execute_process(
    COMMAND "${GIT_EXECUTABLE}" remote get-url origin
    WORKING_DIRECTORY "${RAYD_SOURCE_DIR}"
    RESULT_VARIABLE RAYD_GIT_REMOTE_RESULT
    OUTPUT_VARIABLE RAYD_GIT_REMOTE_URL
    ERROR_VARIABLE RAYD_GIT_REMOTE_ERROR
    OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT RAYD_GIT_REMOTE_RESULT EQUAL 0)
    message(FATAL_ERROR
        "Could not read the RayD origin URL in '${RAYD_SOURCE_DIR}': ${RAYD_GIT_REMOTE_ERROR}")
endif()
if(NOT RAYD_GIT_REMOTE_URL STREQUAL RAYD_LOCK_REPOSITORY_URL)
    message(FATAL_ERROR
        "RayD repository mismatch: origin is '${RAYD_GIT_REMOTE_URL}', "
        "lock requires '${RAYD_LOCK_REPOSITORY_URL}'.")
endif()
set(RAYD_INTEGRATION_ABI_FILE "${RAYD_SOURCE_DIR}/${RAYD_LOCK_ABI_PATH}")
if(NOT EXISTS "${RAYD_INTEGRATION_ABI_FILE}")
    message(FATAL_ERROR "RayD integration ABI header is missing: ${RAYD_LOCK_ABI_PATH}")
endif()
file(SHA256 "${RAYD_INTEGRATION_ABI_FILE}" RAYD_INTEGRATION_ABI_SHA256)
string(TOLOWER "${RAYD_LOCK_ABI_SHA256}" RAYD_LOCK_ABI_SHA256)
if(NOT RAYD_INTEGRATION_ABI_SHA256 STREQUAL RAYD_LOCK_ABI_SHA256)
    message(FATAL_ERROR
        "RayD integration ABI mismatch for ${RAYD_LOCK_ABI_PATH}: "
        "checkout is ${RAYD_INTEGRATION_ABI_SHA256}, "
        "lock requires ${RAYD_LOCK_ABI_SHA256}.")
endif()
execute_process(
    COMMAND "${GIT_EXECUTABLE}" status --porcelain --untracked-files=normal
    WORKING_DIRECTORY "${RAYD_SOURCE_DIR}"
    RESULT_VARIABLE RAYD_GIT_STATUS_RESULT
    OUTPUT_VARIABLE RAYD_GIT_STATUS
    ERROR_VARIABLE RAYD_GIT_STATUS_ERROR
    OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT RAYD_GIT_STATUS_RESULT EQUAL 0)
    message(FATAL_ERROR
        "Could not read the RayD worktree state in '${RAYD_SOURCE_DIR}': ${RAYD_GIT_STATUS_ERROR}")
endif()
if(RAYD_GIT_STATUS STREQUAL "")
    set(RAYD_GIT_DIRTY 0)
else()
    set(RAYD_GIT_DIRTY 1)
    message(WARNING "RayD checkout is dirty; this developer build is not publishable.")
endif()

set(CHANNEL_NATIVE_RELEASE_BUILD_DEFAULT OFF)
if(DEFINED SKBUILD_STATE AND SKBUILD_STATE STREQUAL "wheel")
    set(CHANNEL_NATIVE_RELEASE_BUILD_DEFAULT ON)
endif()
option(
    CHANNEL_NATIVE_RELEASE_BUILD
    "Require clean Channel Native and RayD checkouts suitable for publication."
    ${CHANNEL_NATIVE_RELEASE_BUILD_DEFAULT})
option(
    CHANNEL_NATIVE_VALIDATE_RAYD_ONLY
    "Validate the locked RayD checkout without configuring the CUDA build."
    OFF)
if(CHANNEL_NATIVE_RELEASE_BUILD AND RAYD_GIT_DIRTY)
    message(FATAL_ERROR "CHANNEL_NATIVE_RELEASE_BUILD forbids a dirty RayD checkout.")
endif()
if(CHANNEL_NATIVE_VALIDATE_RAYD_ONLY)
    message(STATUS "Validated locked RayD checkout ${RAYD_GIT_HEAD}.")
    return()
endif()

project(channel_native LANGUAGES CXX CUDA)

if(CMAKE_CONFIGURATION_TYPES)
    message(FATAL_ERROR
        "Channel Native requires a single-config generator so build_type is part "
        "of the native fingerprint; use Ninja with -DCMAKE_BUILD_TYPE=Release.")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type for Channel Native." FORCE)
endif()

set(RAYD_TORCH_BUILD_NATIVE ON CACHE BOOL "Build RayD Torch native core." FORCE)
set(RAYD_TORCH_BUILD_PYTHON_MODULE OFF CACHE BOOL "Do not build RayD's Python module inside Channel Native." FORCE)
add_subdirectory(
    "${RAYD_SOURCE_DIR}/backends/torch"
    "${CMAKE_CURRENT_BINARY_DIR}/rayd-torch"
    EXCLUDE_FROM_ALL)
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
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c
        "import torch; print(torch.__version__.split('+', 1)[0]); print(torch.version.cuda or '')"
    RESULT_VARIABLE CHANNEL_NATIVE_TORCH_IDENTITY_RESULT
    OUTPUT_VARIABLE CHANNEL_NATIVE_TORCH_IDENTITY
    ERROR_VARIABLE CHANNEL_NATIVE_TORCH_IDENTITY_ERROR
    OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT CHANNEL_NATIVE_TORCH_IDENTITY_RESULT EQUAL 0)
    message(FATAL_ERROR
        "Could not inspect the configured PyTorch identity: ${CHANNEL_NATIVE_TORCH_IDENTITY_ERROR}")
endif()
string(REPLACE "\r\n" ";" CHANNEL_NATIVE_TORCH_IDENTITY "${CHANNEL_NATIVE_TORCH_IDENTITY}")
string(REPLACE "\n" ";" CHANNEL_NATIVE_TORCH_IDENTITY "${CHANNEL_NATIVE_TORCH_IDENTITY}")
list(GET CHANNEL_NATIVE_TORCH_IDENTITY 0 CHANNEL_NATIVE_TORCH_VERSION)
list(GET CHANNEL_NATIVE_TORCH_IDENTITY 1 CHANNEL_NATIVE_TORCH_CUDA_VERSION)
if(CHANNEL_NATIVE_TORCH_CUDA_VERSION STREQUAL "")
    message(FATAL_ERROR "Channel Native requires a CUDA-enabled PyTorch build.")
endif()

execute_process(
    COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    RESULT_VARIABLE CHANNEL_NATIVE_GIT_HEAD_RESULT
    OUTPUT_VARIABLE CHANNEL_NATIVE_GIT_SHA
    ERROR_VARIABLE CHANNEL_NATIVE_GIT_HEAD_ERROR
    OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT CHANNEL_NATIVE_GIT_HEAD_RESULT EQUAL 0)
    message(FATAL_ERROR "Could not read the Channel Native Git SHA: ${CHANNEL_NATIVE_GIT_HEAD_ERROR}")
endif()
execute_process(
    COMMAND "${GIT_EXECUTABLE}" status --porcelain --untracked-files=normal
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    RESULT_VARIABLE CHANNEL_NATIVE_GIT_STATUS_RESULT
    OUTPUT_VARIABLE CHANNEL_NATIVE_GIT_STATUS
    ERROR_VARIABLE CHANNEL_NATIVE_GIT_STATUS_ERROR
    OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT CHANNEL_NATIVE_GIT_STATUS_RESULT EQUAL 0)
    message(FATAL_ERROR
        "Could not read the Channel Native worktree state: ${CHANNEL_NATIVE_GIT_STATUS_ERROR}")
endif()
if(CHANNEL_NATIVE_GIT_STATUS STREQUAL "")
    set(CHANNEL_NATIVE_GIT_DIRTY 0)
    set(CHANNEL_NATIVE_GIT_DIRTY_JSON false)
else()
    set(CHANNEL_NATIVE_GIT_DIRTY 1)
    set(CHANNEL_NATIVE_GIT_DIRTY_JSON true)
endif()
if(CHANNEL_NATIVE_RELEASE_BUILD AND CHANNEL_NATIVE_GIT_DIRTY)
    message(FATAL_ERROR "CHANNEL_NATIVE_RELEASE_BUILD forbids a dirty Channel Native checkout.")
endif()

if(RAYD_GIT_DIRTY)
    set(RAYD_GIT_DIRTY_JSON true)
else()
    set(RAYD_GIT_DIRTY_JSON false)
endif()
if(MSVC)
    set(CHANNEL_NATIVE_CXX_ABI "msvc")
elseif(TORCH_CXX_FLAGS MATCHES "_GLIBCXX_USE_CXX11_ABI=1")
    set(CHANNEL_NATIVE_CXX_ABI "cxx11")
elseif(TORCH_CXX_FLAGS MATCHES "_GLIBCXX_USE_CXX11_ABI=0")
    set(CHANNEL_NATIVE_CXX_ABI "pre-cxx11")
else()
    set(CHANNEL_NATIVE_CXX_ABI "unknown")
endif()
set(CHANNEL_NATIVE_BUILD_TYPE "${CMAKE_BUILD_TYPE}")
set(CHANNEL_NATIVE_COMPILER "${CMAKE_CXX_COMPILER_ID}-${CMAKE_CXX_COMPILER_VERSION}")
string(
    REPLACE ";" ","
    CHANNEL_NATIVE_CUDA_ARCHITECTURES_CSV
    "${CHANNEL_NATIVE_CUDA_ARCHITECTURES}")
set(CHANNEL_NATIVE_CUDA_ARCHITECTURES_JSON "")
foreach(CHANNEL_NATIVE_CUDA_ARCH IN LISTS CHANNEL_NATIVE_CUDA_ARCHITECTURES)
    if(NOT CHANNEL_NATIVE_CUDA_ARCHITECTURES_JSON STREQUAL "")
        string(APPEND CHANNEL_NATIVE_CUDA_ARCHITECTURES_JSON ",")
    endif()
    string(APPEND CHANNEL_NATIVE_CUDA_ARCHITECTURES_JSON "\"${CHANNEL_NATIVE_CUDA_ARCH}\"")
endforeach()
set(CHANNEL_NATIVE_BUILD_IDENTITY_JSON
    "{\"build_type\":\"${CHANNEL_NATIVE_BUILD_TYPE}\",\"channel_native_abi_version\":${CHANNEL_NATIVE_ABI_VERSION},\"channel_native_git_dirty\":${CHANNEL_NATIVE_GIT_DIRTY_JSON},\"channel_native_git_sha\":\"${CHANNEL_NATIVE_GIT_SHA}\",\"compiler\":\"${CHANNEL_NATIVE_COMPILER}\",\"cuda_architectures\":[${CHANNEL_NATIVE_CUDA_ARCHITECTURES_JSON}],\"cuda_compiler_version\":\"${CMAKE_CUDA_COMPILER_VERSION}\",\"cuda_version\":\"${CHANNEL_NATIVE_TORCH_CUDA_VERSION}\",\"cxx_abi\":\"${CHANNEL_NATIVE_CXX_ABI}\",\"rayd_commit\":\"${RAYD_GIT_HEAD}\",\"rayd_dirty\":${RAYD_GIT_DIRTY_JSON},\"rayd_integration_abi_kind\":\"${RAYD_LOCK_ABI_KIND}\",\"rayd_integration_abi_path\":\"${RAYD_LOCK_ABI_PATH}\",\"rayd_integration_abi_sha256\":\"${RAYD_LOCK_ABI_SHA256}\",\"rayd_repository_url\":\"${RAYD_LOCK_REPOSITORY_URL}\",\"torch_version\":\"${CHANNEL_NATIVE_TORCH_VERSION}\"}")
string(SHA256 CHANNEL_NATIVE_BUILD_FINGERPRINT "${CHANNEL_NATIVE_BUILD_IDENTITY_JSON}")
set(
    CHANNEL_NATIVE_BUILD_FINGERPRINT_SOURCE
    "${CMAKE_CURRENT_BINARY_DIR}/_channel_native.build-fingerprint")
file(
    WRITE
    "${CHANNEL_NATIVE_BUILD_FINGERPRINT_SOURCE}"
    "${CHANNEL_NATIVE_BUILD_FINGERPRINT}\n")
add_custom_target(
    channel_native_validate_build_identity
    COMMAND
        "${CMAKE_COMMAND}"
        "-DGIT_EXECUTABLE=${GIT_EXECUTABLE}"
        "-DCHANNEL_NATIVE_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
        "-DCHANNEL_NATIVE_EXPECTED_GIT_SHA=${CHANNEL_NATIVE_GIT_SHA}"
        "-DCHANNEL_NATIVE_EXPECTED_GIT_DIRTY=${CHANNEL_NATIVE_GIT_DIRTY}"
        "-DCHANNEL_NATIVE_RAYD_SOURCE_DIR=${RAYD_SOURCE_DIR}"
        "-DCHANNEL_NATIVE_EXPECTED_RAYD_SHA=${RAYD_GIT_HEAD}"
        "-DCHANNEL_NATIVE_EXPECTED_RAYD_DIRTY=${RAYD_GIT_DIRTY}"
        "-DCHANNEL_NATIVE_EXPECTED_RAYD_REMOTE=${RAYD_LOCK_REPOSITORY_URL}"
        "-DCHANNEL_NATIVE_RAYD_ABI_FILE=${RAYD_INTEGRATION_ABI_FILE}"
        "-DCHANNEL_NATIVE_EXPECTED_RAYD_ABI_SHA256=${RAYD_LOCK_ABI_SHA256}"
        "-DCHANNEL_NATIVE_RAYD_LOCK_FILE=${CHANNEL_NATIVE_RAYD_LOCK_FILE}"
        "-DCHANNEL_NATIVE_EXPECTED_RAYD_LOCK_SHA256=${CHANNEL_NATIVE_RAYD_LOCK_SHA256}"
        "-DCHANNEL_NATIVE_RELEASE_BUILD=${CHANNEL_NATIVE_RELEASE_BUILD}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/validate_build_identity.cmake"
    COMMENT "Validating Channel Native and RayD build identity"
    VERBATIM)
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
    native/channel_native/binding/module.cpp
    native/channel_native/binding/build.cpp
    native/channel_native/binding/runtime.cpp
    native/channel_native/binding/materials.cpp
    native/channel_native/binding/fields.cpp
    native/channel_native/binding/rayd.cpp
    native/channel_native/binding/path.cpp
    native/channel_native/binding/montecarlo.cpp
    native/channel_native/binding/bdpt.cpp
    native/channel_native/build_info.cpp
    native/channel_native/bdpt.cpp
    native/channel_native/bdpt_ad_dispatch.cpp
    native/channel_native/rayd/scene.cpp
    native/channel_native/rayd/geometry.cpp
    native/channel_native/rayd/reflection.cpp
    native/channel_native/rayd/diffraction.cpp
    native/channel_native/kernels/accum.cu
    native/channel_native/kernels/bdpt_accum.cu
    native/channel_native/kernels/bdpt_connect_mis.cu
    native/channel_native/kernels/bdpt_connect_samples.cu
    native/channel_native/kernels/bdpt_connect_visibility.cu
    native/channel_native/kernels/bdpt_connect_accumulation.cu
    native/channel_native/kernels/bdpt_connect_ad.cu
    native/channel_native/kernels/bdpt_sampling.cu
    native/channel_native/kernels/bdpt_subpaths.cu
    native/channel_native/kernels/bdpt_subpaths_ad.cu
    native/channel_native/kernels/bdpt_maps.cu
    native/channel_native/kernels/coupled_candidate_capacity.cu
    native/channel_native/kernels/capacity_failure_state.cu
    native/channel_native/kernels/capacity_failure_terminal.cu
    native/channel_native/kernels/coupled_topology.cu
    native/channel_native/kernels/field_transport.cu
    native/channel_native/kernels/field_transport_free_space.cu
    native/channel_native/kernels/field_transport_reflection.cu
    native/channel_native/kernels/field_rough_scale.cu
    native/channel_native/kernels/field_wedge_ad_coupled.cu
    native/channel_native/kernels/field_wedge_ad_project.cu
    native/channel_native/kernels/field_wedge_ad_prepare.cu
    native/channel_native/kernels/diffraction.cu
    native/channel_native/kernels/diffraction_state_capacity.cu
    native/channel_native/kernels/diffraction_path_capacity.cu
    native/channel_native/kernels/diffraction_discovery.cu
    native/channel_native/kernels/deterministic_accum.cu
    native/channel_native/kernels/deterministic_capacity_finalize.cu
    native/channel_native/kernels/enumerated_canonical_capacity_select.cu
    native/channel_native/kernels/evaluated_paths_canonical_capacity_gather.cu
    native/channel_native/kernels/evaluated_paths_canonical_capacity_gather_ad.cu
    native/channel_native/kernels/evaluated_paths_capacity_pack.cu
    native/channel_native/kernels/evaluated_paths_capacity_pack_ad.cu
    native/channel_native/kernels/deterministic_path_table_capacity_pack.cu
    native/channel_native/kernels/deterministic_diffraction_pair_reduce.cu
    native/channel_native/kernels/path_result_capacity_pack.cu
    native/channel_native/kernels/path_result_capacity_pack_ad.cu
    native/channel_native/kernels/deterministic_field.cu
    native/channel_native/kernels/deterministic_topology.cu
    native/channel_native/kernels/los.cu
    native/channel_native/kernels/los_silhouette_clearance.cu
    native/channel_native/kernels/material.cu
    native/channel_native/kernels/path_compaction.cu
    native/channel_native/kernels/reflection_candidate_capacity.cu
    native/channel_native/kernels/path_trace.cu
    native/channel_native/kernels/kirchhoff_table_ad.cu
    native/channel_native/kernels/reflection.cu
    native/channel_native/kernels/scattering.cu
    native/channel_native/kernels/sampling.cu
    native/channel_native/montecarlo_basic.cpp
    native/channel_native/path_solver.cpp
)
set_target_properties(
    _channel_native
    PROPERTIES CUDA_ARCHITECTURES "${CHANNEL_NATIVE_CUDA_ARCHITECTURES}")
add_dependencies(_channel_native channel_native_validate_build_identity)
if(WIN32)
    # CMake 4.3 may append CMAKE_SHARED_MODULE_SUFFIX after the SOABI suffix,
    # producing a non-importable `.pyd.pyd`; sysconfig is authoritative here.
    set_target_properties(
        _channel_native
        PROPERTIES SUFFIX "${CHANNEL_NATIVE_PYTHON_EXTENSION_SUFFIX}")
endif()

target_compile_features(_channel_native PRIVATE cxx_std_17)
# ADR-010 lockstep kernels replicate Torch per-op float rounding. Implicit
# fma contraction would change the rounding of their mul/add chains, so it is
# disabled for these TUs (explicit fmaf calls, e.g. the shared table
# interpolation, are unaffected).
set_source_files_properties(
    native/channel_native/kernels/kirchhoff_table_ad.cu
    PROPERTIES COMPILE_OPTIONS "--fmad=false")
# ADR-030 freezes the source-state-serial additions and the separate
# multiply/add power, VJP, and JVP expressions for the pair reducer.
set_source_files_properties(
    native/channel_native/kernels/deterministic_diffraction_pair_reduce.cu
    PROPERTIES COMPILE_OPTIONS "--fmad=false")
# ADR-029 PathResult packing freezes the former Torch endpoint-angle operation
# order. Contraction across those per-op multiply/add boundaries changes the
# public valid-row bits and its fixed-topology derivative companions.
set_source_files_properties(
    native/channel_native/kernels/path_result_capacity_pack.cu
    native/channel_native/kernels/path_result_capacity_pack_ad.cu
    PROPERTIES COMPILE_OPTIONS "--fmad=false")
target_link_libraries(
    _channel_native
    PRIVATE
        rayd_torch_native_core
        "${TORCH_LIBRARIES}"
        "${TORCH_PYTHON_LIBRARY}"
        "${CHANNEL_NATIVE_PYTHON_LIBRARY}"
        # ADR-015 kirchhoff_table_ad.cu solves the implicit Sinkhorn adjoint
        # with cuSOLVER dense LU (Sgetrf/Sgetrs). The CUDA:: imported targets
        # are provided by the CUDAToolkit package that Torch already discovers
        # (CUDA::cudart is linked by the BUILD_TESTING lockstep target below).
        CUDA::cusolver)
if(WIN32)
    target_link_directories(_channel_native PRIVATE "${CHANNEL_NATIVE_PYTHON_LIBRARY_DIR}")
endif()
target_compile_definitions(_channel_native PRIVATE CHANNEL_NATIVE_WITH_RAYD=1 CHANNEL_NATIVE_WITH_OPTIX=1)
set_source_files_properties(
    native/channel_native/build_info.cpp
    PROPERTIES
    COMPILE_DEFINITIONS
    "CHANNEL_NATIVE_ABI_VERSION=${CHANNEL_NATIVE_ABI_VERSION};CHANNEL_NATIVE_GIT_SHA=\"${CHANNEL_NATIVE_GIT_SHA}\";CHANNEL_NATIVE_GIT_DIRTY=${CHANNEL_NATIVE_GIT_DIRTY};CHANNEL_NATIVE_RAYD_REPOSITORY_URL=\"${RAYD_LOCK_REPOSITORY_URL}\";CHANNEL_NATIVE_RAYD_COMMIT=\"${RAYD_GIT_HEAD}\";CHANNEL_NATIVE_RAYD_DIRTY=${RAYD_GIT_DIRTY};CHANNEL_NATIVE_RAYD_INTEGRATION_ABI_KIND=\"${RAYD_LOCK_ABI_KIND}\";CHANNEL_NATIVE_RAYD_INTEGRATION_ABI_PATH=\"${RAYD_LOCK_ABI_PATH}\";CHANNEL_NATIVE_RAYD_INTEGRATION_ABI_SHA256=\"${RAYD_LOCK_ABI_SHA256}\";CHANNEL_NATIVE_TORCH_VERSION=\"${CHANNEL_NATIVE_TORCH_VERSION}\";CHANNEL_NATIVE_CUDA_VERSION=\"${CHANNEL_NATIVE_TORCH_CUDA_VERSION}\";CHANNEL_NATIVE_CUDA_COMPILER_VERSION=\"${CMAKE_CUDA_COMPILER_VERSION}\";CHANNEL_NATIVE_COMPILER=\"${CHANNEL_NATIVE_COMPILER}\";CHANNEL_NATIVE_CXX_ABI=\"${CHANNEL_NATIVE_CXX_ABI}\";CHANNEL_NATIVE_CUDA_ARCHITECTURES=\"${CHANNEL_NATIVE_CUDA_ARCHITECTURES_CSV}\";CHANNEL_NATIVE_BUILD_TYPE=\"${CHANNEL_NATIVE_BUILD_TYPE}\";CHANNEL_NATIVE_BUILD_FINGERPRINT=\"${CHANNEL_NATIVE_BUILD_FINGERPRINT}\"")
add_custom_command(
    TARGET _channel_native
    POST_BUILD
    COMMAND
        "${CMAKE_COMMAND}" -E copy_if_different
        "${CHANNEL_NATIVE_BUILD_FINGERPRINT_SOURCE}"
        "$<TARGET_FILE_DIR:_channel_native>/_channel_native.build-fingerprint"
    COMMENT "Writing Channel Native build fingerprint sidecar")

install(TARGETS _channel_native LIBRARY DESTINATION witwin/channel_native)
install(
    FILES "${CHANNEL_NATIVE_RAYD_LOCK_FILE}"
    DESTINATION witwin/channel_native/runtime
    RENAME rayd.lock.json)
install(
    FILES "${CHANNEL_NATIVE_BUILD_FINGERPRINT_SOURCE}"
    DESTINATION witwin/channel_native/runtime)

include(CTest)
if(BUILD_TESTING)
    add_executable(
        channel_native_legacy_slab_lockstep
        EXCLUDE_FROM_ALL
        tests/native/legacy_slab_lockstep.cu)
    set_target_properties(
        channel_native_legacy_slab_lockstep
        PROPERTIES CUDA_ARCHITECTURES "${CHANNEL_NATIVE_CUDA_ARCHITECTURES}")
    target_compile_features(channel_native_legacy_slab_lockstep PRIVATE cxx_std_17)
    target_include_directories(
        channel_native_legacy_slab_lockstep
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/native/channel_native"
            "${RAYD_SOURCE_DIR}/shared/include"
            "${RAYD_SOURCE_DIR}/backends/torch/include"
            "${TORCH_INSTALL_PREFIX}/include"
            "${TORCH_INSTALL_PREFIX}/include/torch/csrc/api/include")
    target_link_libraries(channel_native_legacy_slab_lockstep PRIVATE CUDA::cudart)
    add_test(
        NAME channel_native.legacy_slab_lockstep
        COMMAND channel_native_legacy_slab_lockstep)
    set_tests_properties(
        channel_native.legacy_slab_lockstep
        PROPERTIES SKIP_RETURN_CODE 77)
endif()
