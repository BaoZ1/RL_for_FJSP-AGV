cmake_minimum_required(VERSION 3.15...3.29)
project(example LANGUAGES CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++latest")

set(pybind11_DIR ${CMAKE_SOURCE_DIR}/pybind11/build/mock_install/share/cmake/pybind11)
find_package(pybind11 CONFIG REQUIRED)
find_package(Python 3.12 REQUIRED)

pybind11_add_module(graph graph/bind.cpp graph/graph.cpp)
add_custom_command(
    TARGET graph
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_SOURCE_DIR}/build/$<CONFIG>/graph.cp312-win_amd64.pyd
            ${CMAKE_SOURCE_DIR}
)