cmake_minimum_required(VERSION 3.10)

PROJECT(v4l2_stream_exam)

set(CMAKE_SYSTEM_NAME Linux)

set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)


set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -Ofast")

include_directories(
	#${CMAKE_SOURCE_DIR}/inc
	${CMAKE_SOURCE_DIR}/include/libdrm
	${CMAKE_SOURCE_DIR}/include/libdrm/libdrm
	${CMAKE_SOURCE_DIR}/include
)

set(RGA_API_PATH ${CMAKE_SOURCE_DIR}/libs)
set(RGA_API_LIB ${RGA_API_PATH}/librga.so)
set(DRM_API_PATH ${CMAKE_SOURCE_DIR}/libs)
set(DRM_API_LIB ${DRM_API_PATH}/libdrm.so)

file(GLOB_RECURSE SRC_FILES
	${CMAKE_SOURCE_DIR}/src/*.c
	${CMAKE_SOURCE_DIR}/src/*.cpp
)

add_executable(${CMAKE_PROJECT_NAME} 
	 ${SRC_FILES}
)

target_link_libraries(${CMAKE_PROJECT_NAME}
    ${DRM_API_LIB}
    ${RGA_API_LIB}
	#${LIBMPP_LIB}
	#pthread
)

