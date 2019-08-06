# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Minimum CMake required
cmake_minimum_required(VERSION 3.1)

# Project
project(mlperf_bench C CXX)
cmake_policy(SET CMP0069 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)

include(CheckCXXCompilerFlag)
include(CheckLanguage)

set (DOWNLOAD_LOCATION "${CMAKE_CURRENT_BINARY_DIR}/downloads"
CACHE PATH "Location where external projects will be downloaded.")
mark_as_advanced(DOWNLOAD_LOCATION)

# External dependencies
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/external)

if (NOT ORT_ROOT)
  message(FATAL_ERROR " -- ORT_ROOT not set (ie -DORT_ROOT=c:/src/onnxruntime)")
endif()
if (NOT ORT_LIB)
  message(FATAL_ERROR " -- ORT_LIB not set (ie -DORT_LIB=c:/src/onnxruntime/build/Windows/RelWithDebInfo/RelWithDebInfo)")
endif()
link_directories(${ORT_LIB})
message(STATUS "Using ORT from: ${ORT_ROOT}, ${ORT_LIB}")

# Set C++14 as standard for the whole project
set(CMAKE_CXX_STANDARD 17)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Enable CTest
enable_testing()

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Build type not set - using RelWithDebInfo")
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Choose build type: Debug Release RelWithDebInfo." FORCE)
endif()


set(REPO_ROOT ${PROJECT_SOURCE_DIR})

file(GLOB ORT_LIBS
  "${ORT_LIB}/onnxruntime.*"
  "${ORT_LIB}/libonnxruntime.*"  
)
file(COPY ${ORT_LIBS} DESTINATION  ${CMAKE_CURRENT_BINARY_DIR})

find_package(Threads)

if (MSVC)
  #Always enable exception handling, even for Windows ARM
  SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
  #Disable 4100 globally. Too many this kind errors in protobuf
  SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4100")
  SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} /DYNAMICBASE")
else()
  string(APPEND CMAKE_CXX_FLAGS_RELEASE " -march=native -mtune=native")
  string(APPEND CMAKE_C_FLAGS_RELEASE " -march=native -mtune=native")
  string(APPEND CMAKE_CXX_FLAGS_RELWITHDEBINFO " -march=native -mtune=native")
  string(APPEND CMAKE_C_FLAGS_RELWITHDEBINFO " -march=native -mtune=native")
endif()

find_package(PythonInterp 3.5 REQUIRED)
find_package(PythonLibs 3.5 REQUIRED)

# add_subdirectory(${PROJECT_SOURCE_DIR}/external/onnx/third_party/benchmark EXCLUDE_FROM_ALL)

# External dependencies
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/external)

#add_subdirectory(${PROJECT_SOURCE_DIR}/../../loadgen EXCLUDE_FROM_ALL)

get_filename_component(ROOT "${ROOT}" ABSOLUTE)
get_filename_component(REPO_ROOT "${REPO_ROOT}" ABSOLUTE)

include(cxxopts)

include_directories(
  ${REPO_ROOT} 
  ${REPO_ROOT}/include
  ${cxxopts_INCLUDE_DIRS}
  ${ORT_ROOT}/include
)

# bounds checking behavior.
# throw instead of calling terminate if there's a bounds checking violation.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGSL_THROW_ON_CONTRACT_VIOLATION")
# no bounds checking in release build so no perf cost
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DGSL_UNENFORCED_ON_CONTRACT_VIOLATION")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -DGSL_UNENFORCED_ON_CONTRACT_VIOLATION")


#Adjust warning flags
if (WIN32)
    add_definitions(-DPLATFORM_WINDOWS -DNOGDI -DNOMINMAX -D_USE_MATH_DEFINES)
    # parallel build
    # These compiler opitions cannot be forwarded to NVCC, so cannot use add_compiler_options
    string(APPEND CMAKE_CXX_FLAGS " /MP")
    string(APPEND CMAKE_CXX_FLAGS
      " /wd4503" # Decorated name length exceeded.
      " /wd4127" # conditional expression is constant.
      " /wd4146" # unary minus operator applied to unsigned type. Needed for Protobuf
    )
else()
  add_definitions(-DPLATFORM_POSIX)
  # Enable warning and enable space optimization in Linux
  string(APPEND CMAKE_CXX_FLAGS " -Wall -Wextra -ffunction-sections -fdata-sections")
  string(APPEND CMAKE_C_FLAGS " -Wall -Wextra -ffunction-sections -fdata-sections")
  link_libraries(stdc++fs)
endif()

add_subdirectory(
  ../../../loadgen 
  ${PROJECT_BINARY_DIR}/mlperf_loadgen
)

set(EXTERNAL_DEPENDENCIES mlperf_loadgen)
set(EXTERNAL_INCLUDES ${ORT_ROOT}/include ../../../loadgen ${cxxopts_INCLUDE_DIRS})

file(GLOB srcs
  "*.h"
  "*.cc"
)

add_executable(mlperf_bench ${srcs})
target_include_directories(mlperf_bench PRIVATE ${EXTERNAL_INCLUDES})
target_link_libraries(mlperf_bench mlperf_loadgen onnxruntime ${CMAKE_THREAD_LIBS_INIT})
add_dependencies(mlperf_bench cxxopts)
