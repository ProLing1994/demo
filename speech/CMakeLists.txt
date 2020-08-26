cmake_minimum_required(VERSION 3.4)

# use customized find modules
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    "${CMAKE_SOURCE_DIR}/cmake/Modules/")

project(yh_speech)

# 1) add c++ 11 support --std=c++11
# 2) to fix compilation problem: relocation R_X86_64_32 against
#    `.rodata' can not be used when making a shared object;
#    recompile with -fPIC
if (UNIX OR APPLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC")
endif(UNIX OR APPLE)

# CUDA
find_package(CUDA REQUIRED)

# find OPENFST libaray
find_package(OPENFST REQUIRED)

# find PortAudio libaray
find_package(PortAudio REQUIRED)

# find KALDI libaray
find_package(KALDI REQUIRED)

# set default build type
set(CMAKE_BUILD_TYPE Release CACHE STRING "Set default build type to release")

# add 
if (CMAKE_BUILD_TYPE STREQUAL Release)
  add_definitions(-O3)
endif()

# set default debug and release output directory
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)

file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/lib/Release)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/lib/Release)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/lib/Release)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/lib/Release)

file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/lib/Debug)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR}/lib/Debug)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR}/lib/Debug)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR}/lib/Debug)

file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/lib/Dependency)

file(COPY ${MNN_LIBRARIES} DESTINATION ${CMAKE_SOURCE_DIR}/lib/Dependency)

# add submodules
add_subdirectory(deploy)
