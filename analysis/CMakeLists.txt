cmake_minimum_required(VERSION 3.20)
project(subsafe)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(include)

link_libraries(${OpenCV_LIBS})

add_executable(subsafe src/safety.cpp)