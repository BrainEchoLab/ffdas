cmake_minimum_required(VERSION 3.26)
project(ffdas_matlab LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Pre-built libffdas location
if(NOT FFDAS_LIB_DIR)
    message(FATAL_ERROR "FFDAS_LIB_DIR must be set to the directory containing the pre-built libffdas")
endif()
if(NOT FFDAS_INCLUDE_DIR)
    set(FFDAS_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../include")
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

find_package(Matlab REQUIRED COMPONENTS MEX_COMPILER)

include_directories(
    ${FFDAS_INCLUDE_DIR}
    ${Matlab_INCLUDE_DIRS}
    ${Matlab_ROOT_DIR}/toolbox/parallel/gpu/extern/include
)

if(WIN32)
    set(GPUMEX_BINDER_LIB gpumexbinder)
    link_directories(
        ${Matlab_ROOT_DIR}/extern/lib/win64/microsoft
    )
else()
    set(GPUMEX_BINDER_LIB mwgpumexbinder)
endif()

function(add_mex_function target_name source_file)
    matlab_add_mex(
        NAME ${target_name}
        SRC ${source_file}
        OUTPUT_NAME ${target_name}
        LINK_TO ${FFDAS_LIBRARY} ${GPUMEX_BINDER_LIB}
        R2018a
    )
    if(WIN32 AND MSVC)
        target_compile_options(${target_name} PRIVATE
            $<$<CONFIG:Debug>:/MTd>
            $<$<CONFIG:Release>:/MT>
        )
    endif()

    target_link_options(${target_name} PRIVATE
        $<$<NOT:$<BOOL:${WIN32}>>:-static-libstdc++;-static-libgcc>
    )

    set_target_properties(${target_name} PROPERTIES
        INSTALL_RPATH "$ORIGIN"
    )

    install(TARGETS ${target_name}
        RUNTIME DESTINATION "+ffdas/+core"
        LIBRARY DESTINATION "+ffdas/+core"
    )
endfunction()

add_mex_function(ffdas_create
    "${CMAKE_CURRENT_SOURCE_DIR}/+ffdas/+core/src/ffdas_create.cpp")
add_mex_function(ffdas_destroy
    "${CMAKE_CURRENT_SOURCE_DIR}/+ffdas/+core/src/ffdas_destroy.cpp")
add_mex_function(ffdas_device_get
    "${CMAKE_CURRENT_SOURCE_DIR}/+ffdas/+core/src/ffdas_device_get.cpp")
add_mex_function(ffdas_device_set
    "${CMAKE_CURRENT_SOURCE_DIR}/+ffdas/+core/src/ffdas_device_set.cpp")
add_mex_function(ffdas_event_create
    "${CMAKE_CURRENT_SOURCE_DIR}/+ffdas/+core/src/ffdas_event_create.cpp")
add_mex_function(ffdas_event_destroy
    "${CMAKE_CURRENT_SOURCE_DIR}/+ffdas/+core/src/ffdas_event_destroy.cpp")
add_mex_function(ffdas_event_elapsed_time
    "${CMAKE_CURRENT_SOURCE_DIR}/+ffdas/+core/src/ffdas_event_elapsed_time.cpp")
add_mex_function(ffdas_event_record
    "${CMAKE_CURRENT_SOURCE_DIR}/+ffdas/+core/src/ffdas_event_record.cpp")
add_mex_function(ffdas_event_synchronize
    "${CMAKE_CURRENT_SOURCE_DIR}/+ffdas/+core/src/ffdas_event_synchronize.cpp")
add_mex_function(ffdas_gather
    "${CMAKE_CURRENT_SOURCE_DIR}/+ffdas/+core/src/ffdas_gather.cpp")
add_mex_function(ffdas_truncate_rank
    "${CMAKE_CURRENT_SOURCE_DIR}/+ffdas/+core/src/ffdas_truncate_rank.cpp")
add_mex_function(ffdas_interpolate
    "${CMAKE_CURRENT_SOURCE_DIR}/+ffdas/+core/src/ffdas_interpolate.cpp")
add_mex_function(ffdas_das
    "${CMAKE_CURRENT_SOURCE_DIR}/+ffdas/+core/src/ffdas_das.cpp")
add_mex_function(ffdas_das_sparse
    "${CMAKE_CURRENT_SOURCE_DIR}/+ffdas/+core/src/ffdas_das_sparse.cpp")
add_mex_function(ffdas_contraction
    "${CMAKE_CURRENT_SOURCE_DIR}/+ffdas/+core/src/ffdas_contraction.cpp")
add_mex_function(ffdas_greens
    "${CMAKE_CURRENT_SOURCE_DIR}/+ffdas/+core/src/ffdas_greens.cpp")

# Bundle libffdas alongside MEX files (MEX RPATH is $ORIGIN)
get_filename_component(_ffdas_lib_name "${FFDAS_LIBRARY}" NAME)
if(WIN32)
    string(REGEX REPLACE "\\.lib$" ".dll" _ffdas_dll_name "${_ffdas_lib_name}")
    install(FILES "${FFDAS_LIB_DIR}/${_ffdas_dll_name}" DESTINATION "+ffdas/+core")
else()
    install(FILES "${FFDAS_LIBRARY}" DESTINATION "+ffdas/+core")
endif()

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/+ffdas/"
    DESTINATION "+ffdas"
    FILES_MATCHING PATTERN "*.m"
)

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/examples/"
    DESTINATION "examples"
    FILES_MATCHING PATTERN "*.m"
)
