cmake_minimum_required(VERSION 3.16)
project(sentinel LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(onnxruntime REQUIRED)

add_executable(sentinel src/main.cpp)

target_link_libraries(sentinel PRIVATE onnxruntime::onnxruntime)
