cmake_minimum_required(VERSION 3.26)
project(channel_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type for Channel Native." FORCE)
endif()

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

Python_add_library(
    _channel_native
    MODULE
    WITH_SOABI
    native/channel_native/bindings.cpp
    native/channel_native/build_info.cpp
)

target_compile_features(_channel_native PRIVATE cxx_std_17)

install(TARGETS _channel_native LIBRARY DESTINATION witwin/channel_native)
