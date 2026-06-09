cmake_minimum_required(VERSION 3.26)

# Minimum required CUDA architecture: SM 53 for half precision support.
# SM 70+ enables tensor-core-accelerated greens; older SMs use a scalar fallback.
set(MINIMUM_CUDA_ARCH 53)

# Set the CMAKE_CUDA_ARCHITECTURES environment variable to specify the
# target architectures, e.g., CMAKE_CUDA_ARCHITECTURES=75 for SM 75
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES all)
else()
    foreach(arch ${CMAKE_CUDA_ARCHITECTURES})
        if(arch LESS MINIMUM_CUDA_ARCH)
            message(FATAL_ERROR
                "The target CUDA architecture ${arch} is lower than the minimum required SM ${MINIMUM_CUDA_ARCH}")
        endif()
    endforeach()
endif()

project(ffdas VERSION 0.1.0 LANGUAGES CXX CUDA)

option(FFDAS_USE_NVTX "Enable NVTX profiling annotations" ON)
option(FFDAS_BUILD_PYTHON "Build Python bindings" OFF)
option(FFDAS_BUILD_MATLAB "Build MATLAB bindings" OFF)
option(FFDAS_DEVELOPMENT_BUILD "Output binding binaries directly to source tree" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

set(PROJECT_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(SRC_DIR "${PROJECT_ROOT_DIR}/src")

set(FFDAS_SOURCES
    ${SRC_DIR}/core/context.cu
    ${SRC_DIR}/core/type_utils.cu
    ${SRC_DIR}/core/error_utils.cu
    ${SRC_DIR}/core/tensor.cu
    ${SRC_DIR}/das/das.cu
    ${SRC_DIR}/das/das_sparse.cu
    ${SRC_DIR}/das/das_alg1.cu
    ${SRC_DIR}/das/das_alg2.cu
    ${SRC_DIR}/das/das_alg4.cu
    ${SRC_DIR}/greens/greens.cu
    ${SRC_DIR}/truncate_rank/truncate_rank.cu
    ${SRC_DIR}/contraction/contraction.cu
    ${SRC_DIR}/interpolation/interpolation.cu
    ${SRC_DIR}/interpolation/spatial_hash.cu
    ${SRC_DIR}/tensor_ops/contiguous_copy.cu
    ${SRC_DIR}/tensor_ops/gather.cu
    ${SRC_DIR}/tensor_ops/scatter.cu
)

set(FFDAS_OUTPUT_NAME "ffdas_cu${CUDAToolkit_VERSION_MAJOR}")

add_library(ffdas SHARED ${FFDAS_SOURCES})

target_include_directories(ffdas
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_ROOT_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${SRC_DIR}/core
        ${SRC_DIR}/core/math
        ${SRC_DIR}/das
        ${SRC_DIR}/greens
        ${SRC_DIR}/truncate_rank
        ${SRC_DIR}/contraction
        ${SRC_DIR}/interpolation
        ${SRC_DIR}/tensor_ops
)

set_target_properties(ffdas PROPERTIES
    OUTPUT_NAME "${FFDAS_OUTPUT_NAME}"
    POSITION_INDEPENDENT_CODE ON
    CUDA_RUNTIME_LIBRARY Shared
)

target_compile_options(ffdas PRIVATE
    -DFFDAS_BUILDING_LIBRARY
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;--ftz=true;--fmad=true;-O3;--threads;0>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${MSVC}>>:-Xcompiler=/Zc:preprocessor>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:RelWithDebInfo>>:-v;-keep;-Xptxas=-v;-Xptxas=-preserve-relocs;--nvlink-options=-preserve-relocs;-lineinfo>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-v;-keep;-Xptxas=-v;-Xptxas=-preserve-relocs;--nvlink-options=-preserve-relocs;-g;-lineinfo>
)

set(FFDAS_LINK_LIBS CUDA::cuda_driver CUDA::cublas CUDA::cusolver)
if(FFDAS_USE_NVTX)
    target_compile_definitions(ffdas PRIVATE FFDAS_USE_NVTX)
    list(APPEND FFDAS_LINK_LIBS CUDA::nvtx3)
endif()
target_link_libraries(ffdas PRIVATE ${FFDAS_LINK_LIBS})

# Statically link the C++ runtime. The public API is pure C so no C++
# objects cross the boundary. This makes the binary self-contained and
# avoids conflicts with MATLAB's bundled libstdc++.
if(WIN32)
    set_property(TARGET ffdas PROPERTY
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
else()
    target_link_options(ffdas PRIVATE
        $<HOST_LINK:-static-libstdc++;-static-libgcc>)
endif()

message(STATUS "CUDA toolkit: ${CUDAToolkit_VERSION} (major ${CUDAToolkit_VERSION_MAJOR})")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Library name: ${FFDAS_OUTPUT_NAME}")

# Build-tree config export for standalone binding builds or external consumers
export(TARGETS ffdas
    NAMESPACE ffdas::
    FILE "${CMAKE_BINARY_DIR}/ffdasConfig.cmake")
file(WRITE "${CMAKE_BINARY_DIR}/ffdas_cuda_major.txt"
    "${CUDAToolkit_VERSION_MAJOR}")

# Core library install
install(TARGETS ffdas EXPORT ffdasTargets
    LIBRARY DESTINATION lib COMPONENT core
    RUNTIME DESTINATION bin COMPONENT core
    ARCHIVE DESTINATION lib COMPONENT core)
install(FILES include/ffdas.h DESTINATION include COMPONENT core)
install(EXPORT ffdasTargets
    NAMESPACE ffdas::
    DESTINATION lib/cmake/ffdas COMPONENT core)

# Development build: set output directories for binding targets
if(FFDAS_DEVELOPMENT_BUILD)
    if(FFDAS_BUILD_PYTHON)
        set(FFDAS_PYTHON_DEV_DIR
            "${CMAKE_SOURCE_DIR}/bindings/python/ffdas/_core"
            CACHE INTERNAL "")
    endif()
    if(FFDAS_BUILD_MATLAB)
        set(FFDAS_MATLAB_DEV_DIR
            "${CMAKE_SOURCE_DIR}/bindings/matlab/+ffdas/+core"
            CACHE INTERNAL "")
    endif()
endif()

# Bindings
if(FFDAS_BUILD_PYTHON)
    add_subdirectory(bindings/python)
endif()
if(FFDAS_BUILD_MATLAB)
    add_subdirectory(bindings/matlab)
endif()

# Development build: copy shared library to binding source directories
if(FFDAS_DEVELOPMENT_BUILD)
    set(_ffdas_dev_dirs "")
    if(FFDAS_BUILD_PYTHON)
        list(APPEND _ffdas_dev_dirs "${FFDAS_PYTHON_DEV_DIR}")
    endif()
    if(FFDAS_BUILD_MATLAB)
        list(APPEND _ffdas_dev_dirs "${FFDAS_MATLAB_DEV_DIR}")
    endif()
    foreach(_dir IN LISTS _ffdas_dev_dirs)
        add_custom_command(TARGET ffdas POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
                $<TARGET_FILE:ffdas> "${_dir}/")
    endforeach()
endif()
