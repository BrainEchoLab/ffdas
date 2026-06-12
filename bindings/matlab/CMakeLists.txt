cmake_minimum_required(VERSION 3.21)
project(ffdas_matlab LANGUAGES CXX)

find_package(ffdas REQUIRED)
find_package(Matlab REQUIRED COMPONENTS MEX_COMPILER)

include_directories(
    ${Matlab_INCLUDE_DIRS}
    ${Matlab_ROOT_DIR}/toolbox/parallel/gpu/extern/include
)

if(WIN32)
    set(GPUMEX_BINDER_LIB gpumexbinder)
    link_directories(${Matlab_ROOT_DIR}/extern/lib/win64/microsoft)
else()
    set(GPUMEX_BINDER_LIB mwgpumexbinder)
endif()

set(FFDAS_MEX_TARGETS "")

function(add_mex_function target_name source_file)
    matlab_add_mex(
        NAME ${target_name}
        SRC ${source_file}
        OUTPUT_NAME ${target_name}
        LINK_TO ffdas::ffdas ${GPUMEX_BINDER_LIB}
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

    set(FFDAS_MEX_TARGETS ${FFDAS_MEX_TARGETS} ${target_name} PARENT_SCOPE)
endfunction()

set(MEX_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/+ffdas/+core/src")

add_mex_function(ffdas_create "${MEX_SRC_DIR}/ffdas_create.cpp")
add_mex_function(ffdas_destroy "${MEX_SRC_DIR}/ffdas_destroy.cpp")
add_mex_function(ffdas_device_get "${MEX_SRC_DIR}/ffdas_device_get.cpp")
add_mex_function(ffdas_device_set "${MEX_SRC_DIR}/ffdas_device_set.cpp")
add_mex_function(ffdas_device_count "${MEX_SRC_DIR}/ffdas_device_count.cpp")
add_mex_function(ffdas_event_create "${MEX_SRC_DIR}/ffdas_event_create.cpp")
add_mex_function(ffdas_event_destroy "${MEX_SRC_DIR}/ffdas_event_destroy.cpp")
add_mex_function(ffdas_event_elapsed_time "${MEX_SRC_DIR}/ffdas_event_elapsed_time.cpp")
add_mex_function(ffdas_event_record "${MEX_SRC_DIR}/ffdas_event_record.cpp")
add_mex_function(ffdas_event_synchronize "${MEX_SRC_DIR}/ffdas_event_synchronize.cpp")
add_mex_function(ffdas_gather "${MEX_SRC_DIR}/ffdas_gather.cpp")
add_mex_function(ffdas_truncate_rank "${MEX_SRC_DIR}/ffdas_truncate_rank.cpp")
add_mex_function(ffdas_interpolate "${MEX_SRC_DIR}/ffdas_interpolate.cpp")
add_mex_function(ffdas_das "${MEX_SRC_DIR}/ffdas_das.cpp")
add_mex_function(ffdas_das_sparse "${MEX_SRC_DIR}/ffdas_das_sparse.cpp")
add_mex_function(ffdas_contraction "${MEX_SRC_DIR}/ffdas_contraction.cpp")
add_mex_function(ffdas_greens "${MEX_SRC_DIR}/ffdas_greens.cpp")

# Install MEX binaries
foreach(_target IN LISTS FFDAS_MEX_TARGETS)
    install(TARGETS ${_target}
        LIBRARY DESTINATION +ffdas/+core
        RUNTIME DESTINATION +ffdas/+core)
endforeach()

# Install the core shared library alongside MEX files
install(IMPORTED_RUNTIME_ARTIFACTS ffdas::ffdas
    LIBRARY DESTINATION +ffdas/+core
    RUNTIME DESTINATION +ffdas/+core)

# Install MATLAB source files and examples
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/+ffdas/
    DESTINATION +ffdas
    FILES_MATCHING PATTERN "*.m"
    PATTERN "src" EXCLUDE)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/examples/
    DESTINATION examples
    FILES_MATCHING PATTERN "*.m")
