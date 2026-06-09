cmake_minimum_required(VERSION 3.26)

# Minimum required CUDA architecture: SM 53 for half precision support.
# SM 70+ enables tensor-core-accelerated greens; older SMs use a scalar fallback.
set(MINIMUM_CUDA_ARCH 53)

# Set the CMAKE_CUDA_ARCHITECTURES environment variable to specify the 
# target architectures, e.g., CMAKE_CUDA_ARCHITECTURES=75 for SM 75
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES all)
else()
    # If defined, check if it meets the minimum required architecture
    foreach(arch ${CMAKE_CUDA_ARCHITECTURES})
        if(arch LESS MINIMUM_CUDA_ARCH)
            message(FATAL_ERROR
                "The target CUDA architecture ${arch} is lower than the minimum required SM ${MINIMUM_CUDA_ARCH}")
        endif()
    endforeach()
endif()

project(ffdas VERSION 0.1.0 LANGUAGES CXX CUDA)

option(FFDAS_USE_NVTX "Enable NVTX profiling annotations" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

set(PROJECT_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(SRC_DIR "${PROJECT_ROOT_DIR}/src")

include_directories(
    ${PROJECT_ROOT_DIR}/include
    ${SRC_DIR}/core
    ${SRC_DIR}/core/math
    ${SRC_DIR}/das
    ${SRC_DIR}/greens
    ${SRC_DIR}/truncate_rank
    ${SRC_DIR}/contraction
    ${SRC_DIR}/interpolation
    ${SRC_DIR}/tensor_ops
)

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
target_link_libraries(ffdas PUBLIC ${FFDAS_LINK_LIBS})

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

install(TARGETS ffdas
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
    ARCHIVE DESTINATION lib)
install(FILES include/ffdas.h DESTINATION include)
