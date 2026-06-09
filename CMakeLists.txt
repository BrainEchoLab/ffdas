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

project(ffdas LANGUAGES CXX CUDA)

option(BUILD_PYTHON "Build Python bindings" OFF)
option(BUILD_MEX "Build MATLAB (MEX) bindings" OFF)
option(FFDAS_USE_NVTX "Enable NVTX profiling annotations" ON)
option(FETCH_NANOBIND "Automatically fetch nanobind from `https://github.com/wjakob/nanobind.git` if it is not found" OFF)

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

# Objects library used in both shared and static targets
add_library(ffdas_objects OBJECT ${FFDAS_SOURCES})

set_target_properties(ffdas_objects PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_compile_options(ffdas_objects PRIVATE
    -DFFDAS_BUILDING_LIBRARY
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;--ftz=true;--fmad=true;-O3;--threads;0>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${MSVC}>>:-Xcompiler=/Zc:preprocessor>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:RelWithDebInfo>>:-v;-keep;-Xptxas=-v;-Xptxas=-preserve-relocs;--nvlink-options=-preserve-relocs;-lineinfo>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-v;-keep;-Xptxas=-v;-Xptxas=-preserve-relocs;--nvlink-options=-preserve-relocs;-g;-lineinfo>
)

# target_include_directories(ffdas_objects PRIVATE
#     ${SRC_DIR}
#     ${PROJECT_ROOT_DIR}/include
#     ${PROJECT_ROOT_DIR}/include/utils
#     ${PROJECT_ROOT_DIR}/src/include
#     ${PROJECT_ROOT_DIR}/src/include/math
#     ${CUDA_INCLUDE_DIRS}
# )

if(FFDAS_USE_NVTX)
  target_compile_definitions(ffdas_objects PRIVATE FFDAS_USE_NVTX)
  target_link_libraries(ffdas_objects PRIVATE CUDA::nvtx3)
endif()

set(FFDAS_LINK_LIBS CUDA::cuda_driver CUDA::cublas CUDA::cusolver)
if(FFDAS_USE_NVTX)
    list(APPEND FFDAS_LINK_LIBS CUDA::nvtx3)
endif()

# Shared library — dynamic CUDA runtime
add_library(ffdas SHARED $<TARGET_OBJECTS:ffdas_objects>)
target_link_libraries(ffdas PUBLIC ${FFDAS_LINK_LIBS})
set_target_properties(ffdas PROPERTIES
    CUDA_RUNTIME_LIBRARY Shared
)

# Shared library — static CUDA runtime, for embedding in environments that
# bundle their own (potentially outdated) CUDA runtime (e.g. MATLAB)
add_library(ffdas_static_runtime SHARED $<TARGET_OBJECTS:ffdas_objects>)
target_link_libraries(ffdas_static_runtime PUBLIC ${FFDAS_LINK_LIBS})
set_target_properties(ffdas_static_runtime PROPERTIES
    CUDA_RUNTIME_LIBRARY Static
)
target_link_options(ffdas_static_runtime PRIVATE
    $<$<NOT:$<BOOL:${WIN32}>>:-static-libstdc++;-static-libgcc>
)

# Add Python bindings if enabled and dependencies are available
if(BUILD_PYTHON)
    message(STATUS "Python bindings enabled")
    add_subdirectory(bindings/python)
else()
    message(STATUS "Skipping Python bindings compilation")
endif()

# Add MATLAB bindings if enabled and dependencies are available
if(BUILD_MEX)
    message(STATUS "MATLAB bindings enabled")
    add_subdirectory(bindings/matlab)
else()
    message(STATUS "Skipping MEX bindings compilation")
endif()
