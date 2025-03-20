cmake_minimum_required(VERSION 3.15)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++latest")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++2c")

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_core MODULE src/bind.cpp src/FJSP_env.cpp)
install(TARGETS _core DESTINATION ${SKBUILD_PROJECT_NAME})
