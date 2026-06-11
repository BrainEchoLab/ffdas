cmake_minimum_required(VERSION 3.21)
project(ffdas_python LANGUAGES CXX)

find_package(ffdas REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module Development.SABIModule)

find_package(nanobind CONFIG QUIET)
if(NOT nanobind_FOUND)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
    find_package(nanobind CONFIG REQUIRED)
endif()

nanobind_add_module(_ffdas
    "${CMAKE_CURRENT_SOURCE_DIR}/ffdas/_core/src/_ffdas.cpp"
    STABLE_ABI)

target_link_libraries(_ffdas PRIVATE ffdas::ffdas)

if(MSVC)
    target_compile_options(_ffdas PRIVATE /Zc:preprocessor)
endif()

if(UNIX)
    set_target_properties(_ffdas PROPERTIES INSTALL_RPATH "$ORIGIN")
endif()

install(TARGETS _ffdas DESTINATION ffdas/_core)

# Install the core library alongside the extension for development.
# For release wheels, pyproject.toml excludes it — the core library
# ships in its own package (ffdas-core-cu*).
install(IMPORTED_RUNTIME_ARTIFACTS ffdas::ffdas
    LIBRARY DESTINATION ffdas/_core
    RUNTIME DESTINATION ffdas/_core)
