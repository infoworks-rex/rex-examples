cmake_minimum_required(VERSION 3.4.1)

project(led-example)

set(CMAKE_SYSTEM_NAME Linux)

set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -std=c++11")

include_directories(
	${CMAKE_SOURCE_DIR}/inc
)

file(GLOB_RECURSE SRC_FILES
	${CMAKE_SOURCE_DIR}/src/*.c
)

add_executable(${CMAKE_PROJECT_NAME} ${SRC_FILES})
