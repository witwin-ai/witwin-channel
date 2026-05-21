set(WITWIN_CHANNEL_UTILS_NATIVE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
set(WITWIN_CHANNEL_UTILS_KERNELS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../core/kernels")
set(WITWIN_CHANNEL_UTILS_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../core")
set(WITWIN_DETERMINISTIC_KERNELS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../deterministic/kernels")
set(WITWIN_SHARED_NATIVE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../shared")

set(WITWIN_CHANNEL_UTILS_MODULE_SOURCE
    "${WITWIN_CHANNEL_UTILS_NATIVE_ROOT}/module.cpp"
)
set(WITWIN_CHANNEL_UTILS_BINDING_SOURCES
    "${WITWIN_CHANNEL_UTILS_KERNELS_DIR}/shadow_boundary/bind.cpp"
)
set(WITWIN_CHANNEL_UTILS_CUDA_SOURCES
    "${WITWIN_CHANNEL_UTILS_KERNELS_DIR}/shadow_boundary/shadow_boundary.cu"
)

add_library(witwin_channel_utils_native_cuda STATIC ${WITWIN_CHANNEL_UTILS_CUDA_SOURCES})
target_include_directories(witwin_channel_utils_native_cuda
    PUBLIC
        "${WITWIN_SHARED_NATIVE_DIR}"
        "${WITWIN_CHANNEL_UTILS_KERNELS_DIR}"
        "${WITWIN_CHANNEL_UTILS_SOURCE_DIR}"
        "${WITWIN_DETERMINISTIC_KERNELS_DIR}"
)
target_link_libraries(witwin_channel_utils_native_cuda
    PUBLIC
        CUDA::cudart
)
target_compile_features(witwin_channel_utils_native_cuda PUBLIC cxx_std_17)
set_target_properties(witwin_channel_utils_native_cuda PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_RUNTIME_LIBRARY Shared
)

if(MSVC)
    target_compile_options(witwin_channel_utils_native_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/EHsc>
        $<$<COMPILE_LANGUAGE:CXX>:/wd4251>
    )
endif()

nanobind_add_module(
    _channel_utils_native
    NB_DOMAIN drjit
    "${WITWIN_CHANNEL_UTILS_MODULE_SOURCE}"
    ${WITWIN_CHANNEL_UTILS_BINDING_SOURCES}
)
target_include_directories(_channel_utils_native
    PRIVATE
        "${WITWIN_SHARED_NATIVE_DIR}"
        "${WITWIN_CHANNEL_UTILS_KERNELS_DIR}"
        "${WITWIN_CHANNEL_UTILS_SOURCE_DIR}"
        "${WITWIN_DETERMINISTIC_KERNELS_DIR}"
)
target_link_libraries(_channel_utils_native
    PRIVATE
        witwin_channel_utils_native_cuda
        drjit
        drjit-core
        drjit-extra
        nanothread
        CUDA::cudart
)
target_compile_features(_channel_utils_native PRIVATE cxx_std_17)

if(MSVC)
    target_compile_options(_channel_utils_native PRIVATE /EHsc /wd4251)
endif()

install(TARGETS _channel_utils_native
    LIBRARY DESTINATION witwin/channel/_native
    RUNTIME DESTINATION witwin/channel/_native
)
