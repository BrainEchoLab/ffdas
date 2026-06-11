cmake_minimum_required(VERSION 3.26)

# Minimum required CUDA architecture: SM 53 for half precision support.
# SM 70+ enables tensor-core-accelerated greens; older SMs use a scalar fallback.
set(MINIMUM_CUDA_ARCH 53)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
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

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")

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

add_library(ffdas SHARED ${FFDAS_SOURCES})

target_include_directories(ffdas
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
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
    OUTPUT_NAME ffdas
    POSITION_INDEPENDENT_CODE ON
)

set(FFDAS_CUDA_FLAGS --extended-lambda --ftz=true --fmad=true --threads 0)
set(FFDAS_CUDA_DEBUG_FLAGS
    -v -keep
    -Xptxas=-v -Xptxas=-preserve-relocs
    --nvlink-options=-preserve-relocs
    -lineinfo)

target_compile_options(ffdas PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${FFDAS_CUDA_FLAGS}>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<CONFIG:Debug>>>:-O3>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${MSVC}>>:-Xcompiler=/Zc:preprocessor>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>>:${FFDAS_CUDA_DEBUG_FLAGS}>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-g>
)

set(FFDAS_LINK_LIBS CUDA::cudart CUDA::cuda_driver CUDA::cublas CUDA::cusolver)
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

# VS generator: nvcc needs the intermediate directory to exist
if(CMAKE_GENERATOR MATCHES "Visual Studio")
    add_custom_command(TARGET ffdas PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory
            "${CMAKE_BINARY_DIR}/ffdas/${CMAKE_VS_PLATFORM_NAME}/$<CONFIG>")
endif()

message(STATUS "CUDA toolkit: ${CUDAToolkit_VERSION} (major ${CUDAToolkit_VERSION_MAJOR})")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# Build-tree metadata for binding builds and build.py
file(WRITE "${CMAKE_BINARY_DIR}/ffdas_cuda_major.txt"
    "${CUDAToolkit_VERSION_MAJOR}")
file(WRITE "${CMAKE_BINARY_DIR}/ffdas_version.txt"
    "${PROJECT_VERSION}")

# Build-tree config export (used by binding builds via CMAKE_PREFIX_PATH)
export(TARGETS ffdas
    NAMESPACE ffdas::
    FILE "${CMAKE_BINARY_DIR}/ffdasConfig.cmake")

# Install-tree config export
include(GNUInstallDirs)

install(TARGETS ffdas EXPORT ffdasTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES include/ffdas.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT ffdasTargets
    NAMESPACE ffdas::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ffdas)

include(CMakePackageConfigHelpers)
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ffdasConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/ffdasConfig_install.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ffdas)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/ffdasConfigVersion.cmake"
    COMPATIBILITY SameMajorVersion)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/ffdasConfig_install.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ffdas
    RENAME ffdasConfig.cmake)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/ffdasConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ffdas)
