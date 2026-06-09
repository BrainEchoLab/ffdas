cmake_minimum_required(VERSION 3.26)
project(ffdas_python LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Pre-built libffdas location (passed from setup.py or manually)
if(NOT FFDAS_LIB_DIR)
    message(FATAL_ERROR "FFDAS_LIB_DIR must be set to the directory containing the pre-built libffdas")
endif()
if(NOT FFDAS_INCLUDE_DIR)
    message(FATAL_ERROR "FFDAS_INCLUDE_DIR must be set to the directory containing ffdas.h")
endif()

# Find the versioned library (libffdas_cu13.so, ffdas_cu13.lib, etc.)
file(GLOB _ffdas_libs
    "${FFDAS_LIB_DIR}/libffdas_cu*${CMAKE_SHARED_LIBRARY_SUFFIX}"
    "${FFDAS_LIB_DIR}/ffdas_cu*.lib")
list(LENGTH _ffdas_libs _count)
if(_count EQUAL 0)
    message(FATAL_ERROR "No libffdas_cu* found in ${FFDAS_LIB_DIR}")
elseif(_count GREATER 1)
    message(FATAL_ERROR "Multiple libffdas_cu* found in ${FFDAS_LIB_DIR}: ${_ffdas_libs}")
endif()
list(GET _ffdas_libs 0 FFDAS_LIBRARY)
message(STATUS "Found libffdas: ${FFDAS_LIBRARY}")

# CUDA headers needed for vector_types.h and cuda_fp16.h in _ffdas.cpp
find_package(CUDAToolkit REQUIRED)

find_package(Python COMPONENTS Interpreter Development.Module Development.SABIModule REQUIRED)

find_package(nanobind CONFIG QUIET)
if(NOT nanobind_FOUND)
    if(NOT FETCH_NANOBIND)
        message(FATAL_ERROR "nanobind not found. Install via `pip install nanobind` or set FETCH_NANOBIND=ON")
    endif()
    include(FetchContent)
    message(STATUS "Fetching nanobind via FetchContent...")
    FetchContent_Declare(
        nanobind
        GIT_REPOSITORY https://github.com/wjakob/nanobind.git
        GIT_TAG        v2.6.1
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(nanobind)
endif()

nanobind_add_module(_ffdas
    "${CMAKE_CURRENT_SOURCE_DIR}/ffdas/_core/src/_ffdas.cpp"
    STABLE_ABI
)

target_link_libraries(_ffdas PRIVATE ${FFDAS_LIBRARY})
target_include_directories(_ffdas PRIVATE
    ${FFDAS_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

if(MSVC)
    target_compile_options(_ffdas PRIVATE /Zc:preprocessor)
endif()

if(UNIX)
    set_target_properties(_ffdas PROPERTIES
        INSTALL_RPATH "$ORIGIN"
        BUILD_RPATH "${FFDAS_LIB_DIR}"
    )
endif()

# Install the extension and the shared library it depends on
install(TARGETS _ffdas LIBRARY DESTINATION .)
get_filename_component(_ffdas_lib_name "${FFDAS_LIBRARY}" NAME)
if(WIN32)
    # find_library returns .lib on Windows; install the .dll instead
    string(REGEX REPLACE "\\.lib$" ".dll" _ffdas_dll_name "${_ffdas_lib_name}")
    install(FILES "${FFDAS_LIB_DIR}/${_ffdas_dll_name}" DESTINATION .)
else()
    install(FILES "${FFDAS_LIBRARY}" DESTINATION .)
endif()
