set(WITWIN_DETERMINISTIC_RADIOMAP_NATIVE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
set(WITWIN_DETERMINISTIC_RADIOMAP_KERNELS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../deterministic/kernels")
set(WITWIN_DETERMINISTIC_RADIOMAP_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../deterministic")
set(WITWIN_SHARED_NATIVE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../shared")

set(WITWIN_DETERMINISTIC_RADIOMAP_MODULE_SOURCE
    "${WITWIN_DETERMINISTIC_RADIOMAP_NATIVE_ROOT}/module.cpp"
)
set(WITWIN_DETERMINISTIC_RADIOMAP_BINDING_SOURCES
    "${WITWIN_SHARED_NATIVE_DIR}/sample/bind.cpp"
    "${WITWIN_DETERMINISTIC_RADIOMAP_KERNELS_DIR}/utd/bind.cpp"
    "${WITWIN_DETERMINISTIC_RADIOMAP_KERNELS_DIR}/reflection/bind.cpp"
    "${WITWIN_DETERMINISTIC_RADIOMAP_KERNELS_DIR}/packed_state/bind.cpp"
    "${WITWIN_DETERMINISTIC_RADIOMAP_KERNELS_DIR}/radio_map_accumulate/bind.cpp"
    "${WITWIN_DETERMINISTIC_RADIOMAP_KERNELS_DIR}/suffix_grid/bind.cpp"
    "${WITWIN_DETERMINISTIC_RADIOMAP_KERNELS_DIR}/cartesian_filter/bind.cpp"
    "${WITWIN_DETERMINISTIC_RADIOMAP_KERNELS_DIR}/pruning_sort/bind.cpp"
)
set(WITWIN_DETERMINISTIC_RADIOMAP_CUDA_SOURCES
    "${WITWIN_SHARED_NATIVE_DIR}/sample/sample_cuda.cu"
    "${WITWIN_DETERMINISTIC_RADIOMAP_KERNELS_DIR}/utd/utd_accumulate.cu"
    "${WITWIN_DETERMINISTIC_RADIOMAP_KERNELS_DIR}/utd_state_tiles/utd_state_tiles.cu"
    "${WITWIN_DETERMINISTIC_RADIOMAP_KERNELS_DIR}/packed_state/packed_state.cu"
    "${WITWIN_DETERMINISTIC_RADIOMAP_KERNELS_DIR}/reflection/reflection_accumulate.cu"
    "${WITWIN_DETERMINISTIC_RADIOMAP_KERNELS_DIR}/reflection/reflection_prefix_filter.cu"
    "${WITWIN_DETERMINISTIC_RADIOMAP_KERNELS_DIR}/reflection/reflection_prefix_compact.cu"
    "${WITWIN_DETERMINISTIC_RADIOMAP_KERNELS_DIR}/reflection/reflection_jvp.cu"
    "${WITWIN_DETERMINISTIC_RADIOMAP_KERNELS_DIR}/radio_map_accumulate/radio_map_accumulate.cu"
    "${WITWIN_DETERMINISTIC_RADIOMAP_KERNELS_DIR}/suffix_grid/suffix_grid_forward.cu"
    "${WITWIN_DETERMINISTIC_RADIOMAP_KERNELS_DIR}/suffix_grid/suffix_grid_jvp.cu"
    "${WITWIN_DETERMINISTIC_RADIOMAP_KERNELS_DIR}/suffix_grid/suffix_grid_backward.cu"
    "${WITWIN_DETERMINISTIC_RADIOMAP_KERNELS_DIR}/suffix_grid/suffix_segment_tiles.cu"
    "${WITWIN_DETERMINISTIC_RADIOMAP_KERNELS_DIR}/cartesian_filter/cartesian_filter.cu"
    "${WITWIN_DETERMINISTIC_RADIOMAP_KERNELS_DIR}/pruning_sort/pruning_sort.cu"
)

add_library(witwin_deterministic_radiomap_native_cuda STATIC ${WITWIN_DETERMINISTIC_RADIOMAP_CUDA_SOURCES})
target_include_directories(witwin_deterministic_radiomap_native_cuda
    PUBLIC
        "${WITWIN_SHARED_NATIVE_DIR}"
        "${WITWIN_DETERMINISTIC_RADIOMAP_KERNELS_DIR}"
        "${WITWIN_DETERMINISTIC_RADIOMAP_SOURCE_DIR}"
)
target_link_libraries(witwin_deterministic_radiomap_native_cuda
    PUBLIC
        CUDA::cudart
)
target_compile_features(witwin_deterministic_radiomap_native_cuda PUBLIC cxx_std_17)
set_target_properties(witwin_deterministic_radiomap_native_cuda PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_RUNTIME_LIBRARY Shared
)

if(MSVC)
    target_compile_options(witwin_deterministic_radiomap_native_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/EHsc>
        $<$<COMPILE_LANGUAGE:CXX>:/wd4251>
    )
endif()

nanobind_add_module(
    _deterministic_radiomap_native
    NB_DOMAIN drjit
    "${WITWIN_DETERMINISTIC_RADIOMAP_MODULE_SOURCE}"
    ${WITWIN_DETERMINISTIC_RADIOMAP_BINDING_SOURCES}
)
target_include_directories(_deterministic_radiomap_native
    PRIVATE
        "${WITWIN_SHARED_NATIVE_DIR}"
        "${WITWIN_DETERMINISTIC_RADIOMAP_KERNELS_DIR}"
        "${WITWIN_DETERMINISTIC_RADIOMAP_SOURCE_DIR}"
)
target_link_libraries(_deterministic_radiomap_native
    PRIVATE
        witwin_deterministic_radiomap_native_cuda
        drjit
        drjit-core
        drjit-extra
        nanothread
        CUDA::cudart
)
target_compile_features(_deterministic_radiomap_native PRIVATE cxx_std_17)

if(MSVC)
    target_compile_options(_deterministic_radiomap_native PRIVATE /EHsc /wd4251)
endif()

install(TARGETS _deterministic_radiomap_native
    LIBRARY DESTINATION witwin/channel/_native
    RUNTIME DESTINATION witwin/channel/_native
)
