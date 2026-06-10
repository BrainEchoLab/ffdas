cmake_minimum_required(VERSION 3.26)

# Minimum required CUDA architecture: SM 53 for half precision support.
# SM 70+ enables tensor-core-accelerated greens; older SMs use a scalar fallback.
set(MINIMUM_CUDA_ARCH 53)

# Set the CMAKE_CUDA_ARCHITECTURES environment variable to specify the
# target architectures, e.g., CMAKE_CUDA_ARCHITECTURES=75 for SM 75
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

# NVIDIA's MSBuild CUDA targets use --keep-dir set to a per-target
# intermediate directory but don't create it. Without this, nvcc fails
# with "Failed to create the host compiler response file".
if(CMAKE_GENERATOR MATCHES "Visual Studio")
    add_custom_command(TARGET ffdas PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory
            "${CMAKE_BINARY_DIR}/ffdas/${CMAKE_VS_PLATFORM_NAME}/$<CONFIG>")
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
    file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/ffdas_python_files.txt"
        CONTENT "$<TARGET_FILE:_ffdas>\n$<TARGET_FILE:ffdas>\n")
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

if(FFDAS_BUILD_PYTHON AND FFDAS_BUILD_MATLAB)
    find_package(Python COMPONENTS Interpreter)

    set(FFDAS_DIST_DIR "${CMAKE_BINARY_DIR}/dist" CACHE PATH
        "Output directory for release artifacts")

    set(_cu "cu${CUDAToolkit_VERSION_MAJOR}")
    if(WIN32)
        set(_plat "win_amd64")
    else()
        set(_plat "linux_x86_64")
    endif()

    set(_wheel "ffdas_${_cu}-${PROJECT_VERSION}-cp312-abi3-${_plat}.whl")

    message("${_wheel}")
    message("${FFDAS_DIST_DIR}")
    message("${CMAKE_BINARY_DIR}")

    if(WIN32)
        add_custom_target(wheel
            COMMAND ${CMAKE_COMMAND} -E make_directory "${FFDAS_DIST_DIR}"
            COMMAND ${CMAKE_COMMAND} -E env "FFDAS_BUILD_DIR=${CMAKE_BINARY_DIR}"
                "$<TARGET_FILE:Python::Interpreter>" -m build --wheel
                --outdir "${FFDAS_DIST_DIR}"
                "${CMAKE_SOURCE_DIR}/bindings/python"
            COMMAND delvewheel repair "${FFDAS_DIST_DIR}/${_wheel}"
                --wheel-dir "${FFDAS_DIST_DIR}"
                --add-path "${CMAKE_BINARY_DIR}"
                --exclude "cublas64_*.dll"
                --exclude "cublasLt64_*.dll"
                --exclude "cusolver64_*.dll"
                --exclude "cusparse64_*.dll"
            DEPENDS ffdas _ffdas
            VERBATIM
        )
    else()
        add_custom_target(wheel
            COMMAND ${CMAKE_COMMAND} -E make_directory "${FFDAS_DIST_DIR}"
            COMMAND ${CMAKE_COMMAND} -E env "FFDAS_BUILD_DIR=${CMAKE_BINARY_DIR}"
                "$<TARGET_FILE:Python::Interpreter>" -m build --wheel
                --outdir "${FFDAS_DIST_DIR}"
                "${CMAKE_SOURCE_DIR}/bindings/python"
            COMMAND auditwheel repair "${FFDAS_DIST_DIR}/${_wheel}"
                --wheel-dir "${FFDAS_DIST_DIR}"
                --exclude "libcudart.so.*"
                --exclude "libcublas.so.*"
                --exclude "libcusolver.so.*"
                --exclude "libcublasLt.so.*"
                --exclude "libcusparse.so.*"
                --exclude "libnvToolsExt.so.*"
                --exclude "libcuda.so.*"
            COMMAND ${CMAKE_COMMAND} -E rm "${FFDAS_DIST_DIR}/${_wheel}"
            DEPENDS ffdas _ffdas
            VERBATIM
        )
    endif()

    set(_matlab "ffdas_${_cu}-${PROJECT_VERSION}-matlab-${_plat}")

    add_custom_target(matlab_package
        COMMAND ${CMAKE_COMMAND} -E make_directory "${FFDAS_DIST_DIR}"
        COMMAND ${CMAKE_COMMAND} --install "${CMAKE_BINARY_DIR}"
            --component matlab --prefix "${FFDAS_DIST_DIR}/${_matlab}"
        COMMAND ${CMAKE_COMMAND} -E chdir "${FFDAS_DIST_DIR}"
            ${CMAKE_COMMAND} -E tar cf "${_matlab}.zip" --format=zip -- "${_matlab}"
        COMMAND ${CMAKE_COMMAND} -E rm -rf "${FFDAS_DIST_DIR}/${_matlab}"
        DEPENDS ffdas ffdas_matlab_all
        VERBATIM
    )

    add_custom_target(release DEPENDS wheel matlab_package)
endif()


# # linux
# CUDACXX=/usr/local/cuda-12/bin/nvcc CMAKE_INSTALL_PREFIX=dist cmake -S . -B _build -DCMAKE_BUILD_TYPE=Release -DFFDAS_BUILD_PYTHON=ON -DFFDAS_BUILD_MATLAB=ON -DFETCH_NANOBIND=ON -DCMAKE_CUDA_ARCHITECTURES="53-real;70-real;75-real;80-real;86-real;89-real;90" && cmake --build _build --target install
# cmake --build _build
# python -m build --wheel --outdir dist bindings/python
# auditwheel repair dist/ffdas_cu12-0.1.0-cp312-abi3-linux_x86_64.whl --wheel-dir dist/ --exclude libcudart.so.* --exclude libcublas.so.* --exclude libcusolver.so.* --exclude libcublasLt.so.* --exclude libcusparse.so.* --exclude libnvToolsExt.so.* --exclude libcuda.so.*
# cmake --install _build --component matlab --prefix dist/ffdas_cu12-0.1.0-matlab-linux_x86_64/
# cd dist
# zip -r ffdas_cu12-0.1.0-matlab-linux_x86_64.zip ffdas_cu12-0.1.0-matlab-linux_x86_64/
# rm -r ffdas_cu12-0.1.0-matlab-linux_x86_64/

# cmake -S . -B _build -DCMAKE_BUILD_TYPE=Release -DFFDAS_BUILD_PYTHON=ON -DFFDAS_BUILD_MATLAB=ON -DCUDACXX=/usr/local/cuda-13/bin/nvcc -DFETCH_NANOBIND=ON -DCMAKE_CUDA_ARCHITECTURES="75-real;80-real;86-real;89-real;90-real;100-real;120"
# cmake --build _build
# python -m build --wheel --outdir dist bindings/python
# auditwheel repair dist/ffdas_cu13-0.1.0-cp312-abi3-linux_x86_64.whl --wheel-dir dist/ --exclude libcudart.so.* --exclude libcublas.so.* --exclude libcusolver.so.* --exclude libcublasLt.so.* --exclude libcusparse.so.* --exclude libnvToolsExt.so.* --exclude libcuda.so.*
# cmake --install _build --component matlab --prefix dist/ffdas_cu13-0.1.0-matlab-linux_x86_64/
# cd dist
# zip -r ffdas_cu13-0.1.0-matlab-linux_x86_64.zip ffdas_cu13-0.1.0-matlab-linux_x86_64/
# rm -r ffdas_cu13-0.1.0-matlab-linux_x86_64/

# # windows
# $vsPath = & "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe" -latest -property installationPath
# Import-Module "$vsPath\Common7\Tools\Microsoft.VisualStudio.DevShell.dll"
# Enter-VsDevShell -VsInstallPath $vsPath -Arch amd64 -SkipAutomaticLocation

# cmake -S . -B _build -G Ninja -DCMAKE_BUILD_TYPE=Release -DFFDAS_BUILD_PYTHON=ON -DFFDAS_BUILD_MATLAB=ON -DCUDACXX="c:/program files/nvidia gpu computing toolkit/cuda/v13.0/bin/nvcc.exe" -DFETCH_NANOBIND=ON -DCMAKE_CUDA_ARCHITECTURES="75-real;80-real;86-real;89-real;90-real;100-real;120"
# cmake --build _build
# python -m build --wheel --outdir dist bindings/python
# delvewheel repair dist/ffdas_cu13-0.1.0-cp312-abi3-win_amd64.whl --wheel-dir dist/ --add-path _build --exclude cublas64_*.dll --exclude cublasLt64_*.dll --exclude cusolver64_*.dll --exclude cusparse64_*.dll
# cmake --install _build --component matlab --prefix dist/ffdas_cu13-0.1.0-matlab-win_amd64/ 
# Compress-Archive -Path dist/ffdas_cu13-0.1.0-matlab-win_amd64/ -DestinationPath dist/ffdas_cu13-0.1.0-matlab-win_amd64.zip
# Remove-Item -Recurse dist/ffdas_cu13-0.1.0-matlab-win_amd64

# cmake -S . -B _build -G Ninja -DCMAKE_BUILD_TYPE=Release -DFFDAS_BUILD_PYTHON=ON -DFFDAS_BUILD_MATLAB=ON -DCUDACXX="c:/program files/nvidia gpu computing toolkit/cuda/v12.5/bin/nvcc.exe" -DFETCH_NANOBIND=ON -DCMAKE_CUDA_ARCHITECTURES="53-real;70-real;75-real;80-real;86-real;89-real;90"
# cmake --build _build
# python -m build --wheel --outdir dist bindings/python
# delvewheel repair dist/ffdas_cu12-0.1.0-cp312-abi3-win_amd64.whl --wheel-dir dist/ --add-path _build --exclude cublas64_*.dll --exclude cublasLt64_*.dll --exclude cusolver64_*.dll --exclude cusparse64_*.dll
# cmake --install _build --component matlab --prefix dist/ffdas_cu12-0.1.0-matlab-win_amd64/ 
# Compress-Archive -Path dist/ffdas_cu12-0.1.0-matlab-win_amd64/ -DestinationPath dist/ffdas_cu12-0.1.0-matlab-win_amd64.zip
# Remove-Item -Recurse dist/ffdas_cu12-0.1.0-matlab-win_amd64

# CUDACXX=/usr/local/cuda-12/bin/nvcc cmake -S . -B _build_cu12 -DCMAKE_BUILD_TYPE=Release -DFFDAS_BUILD_PYTHON=ON -DFFDAS_BUILD_MATLAB=ON -DFETCH_NANOBIND=ON -DCMAKE_CUDA_ARCHITECTURES="53-real;70-real;75-real;80-real;86-real;89-real;90"
# cmake --build _build_cu12 --target release
