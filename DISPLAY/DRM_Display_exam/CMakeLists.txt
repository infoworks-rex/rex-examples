cmake_minimum_required(VERSION 3.4.1)

project(drm_display_exam)

set(CMAKE_SYSTEM_NAME Linux)

set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -std=c++11 -O3 -Ofast")

include_directories(
	${CMAKE_SOURCE_DIR}/include/libdrm/libdrm
	${CMAKE_SOURCE_DIR}/include/opencv
	${CMAKE_SOURCE_DIR}/include
)

link_directories(drm_display_exam
	${CMAKE_SOURCE_DIR}/libs/libdrm
	${CMAKE_SOURCE_DIR}/libs/opencv
	${CMAKE_SOURCE_DIR}/libs/rga
)

file(GLOB_RECURSE SRC_FILES
	${CMAKE_SOURCE_DIR}/src/*.cc

)

add_executable(drm_display_exam
	${SRC_FILES}
)

target_link_libraries(drm_display_exam 
	drm m rga opencv_core opencv_highgui opencv_imgcodecs opencv_imgproc
)
