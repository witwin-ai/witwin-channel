set(WITWIN_MC_RADIOMAP_NATIVE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
set(WITWIN_MC_RADIOMAP_KERNELS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../montecarlo/kernels")
set(WITWIN_MC_RADIOMAP_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../montecarlo")
set(WITWIN_SHARED_NATIVE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../shared")

set(WITWIN_MC_RADIOMAP_MODULE_SOURCE
    "${WITWIN_MC_RADIOMAP_NATIVE_ROOT}/module.cpp"
)
set(WITWIN_MC_RADIOMAP_BINDING_SOURCES
    "${WITWIN_MC_RADIOMAP_KERNELS_DIR}/sparse_coeff/bind.cpp"
    "${WITWIN_MC_RADIOMAP_KERNELS_DIR}/transport_grid/bind.cpp"
    "${WITWIN_MC_RADIOMAP_KERNELS_DIR}/transport_vertex/bind.cpp"
    "${WITWIN_MC_RADIOMAP_KERNELS_DIR}/diffraction_builder/bind.cpp"
)
set(WITWIN_MC_RADIOMAP_CUDA_SOURCES
    "${WITWIN_MC_RADIOMAP_KERNELS_DIR}/sparse_coeff/monte_carlo_sparse_coeff.cu"
    "${WITWIN_MC_RADIOMAP_KERNELS_DIR}/transport_grid/transport_grid.cu"
    "${WITWIN_MC_RADIOMAP_KERNELS_DIR}/transport_vertex/transport_vertex.cu"
    "${WITWIN_MC_RADIOMAP_KERNELS_DIR}/diffraction_builder/diffraction_builder.cu"
)

add_library(witwin_monte_carlo_radiomap_native_cuda STATIC ${WITWIN_MC_RADIOMAP_CUDA_SOURCES})
target_include_directories(witwin_monte_carlo_radiomap_native_cuda
    PUBLIC
        "${WITWIN_SHARED_NATIVE_DIR}"
        "${WITWIN_MC_RADIOMAP_KERNELS_DIR}"
        "${WITWIN_MC_RADIOMAP_SOURCE_DIR}"
)
target_link_libraries(witwin_monte_carlo_radiomap_native_cuda
    PUBLIC
        CUDA::cudart
)
target_compile_features(witwin_monte_carlo_radiomap_native_cuda PUBLIC cxx_std_17)
set_target_properties(witwin_monte_carlo_radiomap_native_cuda PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_RUNTIME_LIBRARY Shared
)

if(MSVC)
    target_compile_options(witwin_monte_carlo_radiomap_native_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/EHsc>
        $<$<COMPILE_LANGUAGE:CXX>:/wd4251>
    )
endif()

nanobind_add_module(
    _monte_carlo_radiomap_native
    NB_DOMAIN drjit
    "${WITWIN_MC_RADIOMAP_MODULE_SOURCE}"
    ${WITWIN_MC_RADIOMAP_BINDING_SOURCES}
)
target_include_directories(_monte_carlo_radiomap_native
    PRIVATE
        "${WITWIN_SHARED_NATIVE_DIR}"
        "${WITWIN_MC_RADIOMAP_KERNELS_DIR}"
        "${WITWIN_MC_RADIOMAP_SOURCE_DIR}"
)
target_link_libraries(_monte_carlo_radiomap_native
    PRIVATE
        witwin_monte_carlo_radiomap_native_cuda
        drjit
        drjit-core
        drjit-extra
        nanothread
        CUDA::cudart
)
target_compile_features(_monte_carlo_radiomap_native PRIVATE cxx_std_17)

if(MSVC)
    target_compile_options(_monte_carlo_radiomap_native PRIVATE /EHsc /wd4251)
endif()

install(TARGETS _monte_carlo_radiomap_native
    LIBRARY DESTINATION witwin/channel/_native
    RUNTIME DESTINATION witwin/channel/_native
)
