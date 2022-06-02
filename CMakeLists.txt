cmake_minimum_required(VERSION 3.22)
project(wpimath)

set(CMAKE_CXX_STANDARD 17)

include_directories("src/main/native/include")
include_directories("src/main/native/autodiff/include")
include_directories ("src/main/native/eigeninclude")

add_executable(wpimath src/test/native/cpp/optimization/main.cpp)
