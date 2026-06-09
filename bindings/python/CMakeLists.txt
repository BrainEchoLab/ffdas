find_package(Python COMPONENTS Interpreter Development.Module Development.SABIModule REQUIRED)

find_package(nanobind CONFIG QUIET)

if(NOT nanobind_FOUND)
    if (NOT FETCH_NANOBIND)
        message(FATAL_ERROR "nanobind could not be found and FETCH_NANOBIND is disabled. Install it via `python -m pip install nanobind` or define FETCH_NANOBIND=ON")
    endif()

    include(FetchContent)

    message(STATUS "nanobind not found, fetching via FetchContent...")

    include(FetchContent)
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

target_link_libraries(_ffdas PRIVATE ffdas)
target_include_directories(_ffdas PRIVATE
    ${PROJECT_ROOT_DIR}/include
)

if(MSVC)
    target_compile_options(_ffdas PRIVATE /Zc:preprocessor)
endif()

if(UNIX)
    set_target_properties(_ffdas PROPERTIES
        INSTALL_RPATH "$ORIGIN"
        BUILD_RPATH "$ORIGIN"
    )
endif()

install(TARGETS _ffdas
    DESTINATION _core
)
