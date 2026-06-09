find_package(Python COMPONENTS Interpreter Development.Module Development.SABIModule REQUIRED)

find_package(nanobind CONFIG QUIET)
if(NOT nanobind_FOUND)
    option(FETCH_NANOBIND "Fetch nanobind via FetchContent" OFF)
    if(NOT FETCH_NANOBIND)
        message(FATAL_ERROR
            "nanobind not found. Install via pip or set FETCH_NANOBIND=ON")
    endif()
    include(FetchContent)
    FetchContent_Declare(nanobind
        GIT_REPOSITORY https://github.com/wjakob/nanobind.git
        GIT_TAG v2.6.1
        GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(nanobind)
endif()

nanobind_add_module(_ffdas
    "${CMAKE_CURRENT_SOURCE_DIR}/ffdas/_core/src/_ffdas.cpp"
    STABLE_ABI)

target_link_libraries(_ffdas PRIVATE ffdas)

if(MSVC)
    target_compile_options(_ffdas PRIVATE /Zc:preprocessor)
endif()

if(UNIX)
    set_target_properties(_ffdas PROPERTIES INSTALL_RPATH "$ORIGIN")
endif()

if(FFDAS_DEVELOPMENT_BUILD)
    set_target_properties(_ffdas PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${FFDAS_PYTHON_DEV_DIR}"
        RUNTIME_OUTPUT_DIRECTORY "${FFDAS_PYTHON_DEV_DIR}")
endif()

# Install component for packaging
install(TARGETS _ffdas
    LIBRARY DESTINATION ffdas/_core
    COMPONENT python)
install(TARGETS ffdas
    LIBRARY DESTINATION ffdas/_core COMPONENT python
    RUNTIME DESTINATION ffdas/_core COMPONENT python)
