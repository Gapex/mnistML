cmake_minimum_required(VERSION 3.1)
set(CMAKE_BUILD_TYPE "Debug")
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
project(minist-machine-learning)
include_directories(includes)
add_subdirectory(src)