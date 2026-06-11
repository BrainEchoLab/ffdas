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

set_target_properties(_ffdas PROPERTIES INSTALL_RPATH "")

install(TARGETS _ffdas DESTINATION ffdas/_core)
