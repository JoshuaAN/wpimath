cmake_minimum_required(VERSION 3.22)
project(wpimath)

set(CMAKE_CXX_STANDARD 17)

#include(FetchContent)
#FetchContent_Declare(
#        googletest
#        URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
#)

include_directories("src/main/native/include")
include_directories("src/main/native/autodiff/include")
include_directories ("src/main/native/eigeninclude")

add_executable(wpimath src/test/native/cpp/optimization/main.cpp
        src/main/native/cpp/optimization/AutodiffWrapper.cpp
        src/main/native/cpp/optimization/EqualityConstraint.cpp
        src/main/native/cpp/optimization/InequalityConstraint.cpp
        src/main/native/cpp/optimization/Problem.cpp
        src/main/native/cpp/optimization/Variable.cpp
        src/main/native/include/frc/optimization/Variable.h
        )
