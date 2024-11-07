cmake_minimum_required(VERSION 3.15...3.29)
project(example LANGUAGES CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++latest")

set(pybind11_DIR ${CMAKE_SOURCE_DIR}/pybind11/build/mock_install/share/cmake/pybind11)
find_package(pybind11 CONFIG REQUIRED)
find_package(Python 3.12 REQUIRED)

pybind11_add_module(FJSP_env FJSP_env/bind.cpp FJSP_env/FJSP_env.cpp)
