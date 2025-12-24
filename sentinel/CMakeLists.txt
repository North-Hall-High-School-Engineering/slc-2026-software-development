cmake_minimum_required(VERSION 3.16)
project(sentinel LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(onnxruntime REQUIRED)
find_package(ZeroMQ REQUIRED)
find_package(cppzmq REQUIRED)

add_executable(sentinel src/main.cpp)

target_link_libraries(sentinel PRIVATE onnxruntime::onnxruntime libzmq-static cppzmq)
