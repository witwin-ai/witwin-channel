cmake_minimum_required(VERSION 3.26)

foreach(required_variable IN ITEMS
        GIT_EXECUTABLE
        CHANNEL_NATIVE_SOURCE_DIR
        CHANNEL_NATIVE_EXPECTED_GIT_SHA
        CHANNEL_NATIVE_EXPECTED_GIT_DIRTY
        CHANNEL_NATIVE_RAYD_SOURCE_DIR
        CHANNEL_NATIVE_RAYD_SOURCE_KIND
        CHANNEL_NATIVE_EXPECTED_RAYD_SHA
        CHANNEL_NATIVE_EXPECTED_RAYD_DIRTY
        CHANNEL_NATIVE_EXPECTED_RAYD_REMOTE
        CHANNEL_NATIVE_RAYD_ABI_FILE
        CHANNEL_NATIVE_EXPECTED_RAYD_ABI_SHA256
        CHANNEL_NATIVE_RAYD_LOCK_FILE
        CHANNEL_NATIVE_EXPECTED_RAYD_LOCK_SHA256
        CHANNEL_NATIVE_PYTHON_EXECUTABLE
        CHANNEL_NATIVE_RAYD_RESOLVER
        CHANNEL_NATIVE_RELEASE_BUILD)
    if(NOT DEFINED ${required_variable})
        message(FATAL_ERROR "Build identity validation is missing ${required_variable}.")
    endif()
endforeach()

function(read_git_value source_directory output_variable)
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" ${ARGN}
        WORKING_DIRECTORY "${source_directory}"
        RESULT_VARIABLE command_result
        OUTPUT_VARIABLE command_output
        ERROR_VARIABLE command_error
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT command_result EQUAL 0)
        message(FATAL_ERROR
            "Build identity validation failed in '${source_directory}': ${command_error}")
    endif()
    set(${output_variable} "${command_output}" PARENT_SCOPE)
endfunction()

function(validate_git_checkout
        label
        source_directory
        expected_sha
        expected_dirty)
    read_git_value("${source_directory}" actual_sha rev-parse HEAD)
    if(NOT actual_sha STREQUAL expected_sha)
        message(FATAL_ERROR
            "${label} Git HEAD changed after configure: expected ${expected_sha}, "
            "found ${actual_sha}; rerun CMake configure.")
    endif()
    read_git_value(
        "${source_directory}"
        status_output
        status --porcelain --untracked-files=normal)
    if(status_output STREQUAL "")
        set(actual_dirty 0)
    else()
        set(actual_dirty 1)
    endif()
    if(CHANNEL_NATIVE_RELEASE_BUILD AND actual_dirty)
        message(FATAL_ERROR "Release build forbids a dirty ${label} checkout.")
    endif()
    if(NOT actual_dirty EQUAL expected_dirty)
        message(FATAL_ERROR
            "${label} dirty state changed after configure; rerun CMake configure.")
    endif()
endfunction()

validate_git_checkout(
    "Channel Native"
    "${CHANNEL_NATIVE_SOURCE_DIR}"
    "${CHANNEL_NATIVE_EXPECTED_GIT_SHA}"
    "${CHANNEL_NATIVE_EXPECTED_GIT_DIRTY}")
if(CHANNEL_NATIVE_RAYD_SOURCE_KIND STREQUAL "git-checkout")
    validate_git_checkout(
        "RayD"
        "${CHANNEL_NATIVE_RAYD_SOURCE_DIR}"
        "${CHANNEL_NATIVE_EXPECTED_RAYD_SHA}"
        "${CHANNEL_NATIVE_EXPECTED_RAYD_DIRTY}")
    read_git_value(
        "${CHANNEL_NATIVE_RAYD_SOURCE_DIR}"
        actual_rayd_remote
        remote get-url origin)
    if(NOT actual_rayd_remote STREQUAL CHANNEL_NATIVE_EXPECTED_RAYD_REMOTE)
        message(FATAL_ERROR
            "RayD origin changed after configure: expected "
            "'${CHANNEL_NATIVE_EXPECTED_RAYD_REMOTE}', found '${actual_rayd_remote}'; "
            "rerun CMake configure.")
    endif()
elseif(CHANNEL_NATIVE_RAYD_SOURCE_KIND STREQUAL "python-package")
    execute_process(
        COMMAND
            "${CHANNEL_NATIVE_PYTHON_EXECUTABLE}"
            "${CHANNEL_NATIVE_RAYD_RESOLVER}"
            --lock "${CHANNEL_NATIVE_RAYD_LOCK_FILE}"
            --expected-source "${CHANNEL_NATIVE_RAYD_SOURCE_DIR}"
        RESULT_VARIABLE package_validation_result
        ERROR_VARIABLE package_validation_error)
    if(NOT package_validation_result EQUAL 0)
        message(FATAL_ERROR
            "RayD package source changed after configure: ${package_validation_error}")
    endif()
else()
    message(FATAL_ERROR
        "Unknown RayD source kind '${CHANNEL_NATIVE_RAYD_SOURCE_KIND}'.")
endif()

file(SHA256 "${CHANNEL_NATIVE_RAYD_ABI_FILE}" actual_rayd_abi_sha256)
if(NOT actual_rayd_abi_sha256 STREQUAL CHANNEL_NATIVE_EXPECTED_RAYD_ABI_SHA256)
    message(FATAL_ERROR
        "RayD integration ABI changed after configure; rerun CMake configure.")
endif()

file(SHA256 "${CHANNEL_NATIVE_RAYD_LOCK_FILE}" actual_rayd_lock_sha256)
if(NOT actual_rayd_lock_sha256 STREQUAL CHANNEL_NATIVE_EXPECTED_RAYD_LOCK_SHA256)
    message(FATAL_ERROR "RayD lock changed after configure; rerun CMake configure.")
endif()
