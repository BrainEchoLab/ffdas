find_package(Matlab REQUIRED COMPONENTS MEX_COMPILER)

include_directories(
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
        LINK_TO ffdas_static_runtime ${GPUMEX_BINDER_LIB}
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
add_mex_function(ffdas_eigfilter
    "${CMAKE_CURRENT_SOURCE_DIR}/+ffdas/+core/src/ffdas_eigfilter.cpp")
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

# Install rules for packaging
install(TARGETS ffdas_static_runtime
    RUNTIME DESTINATION "+ffdas/+core"
    LIBRARY DESTINATION "+ffdas/+core"
)

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/+ffdas/"
    DESTINATION "+ffdas"
    FILES_MATCHING PATTERN "*.m"
)

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/examples/"
    DESTINATION "examples"
    FILES_MATCHING PATTERN "*.m"
)
