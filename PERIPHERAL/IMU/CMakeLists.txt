cmake_minimum_required(VERSION 3.4.1)

project(imu_exam)

set(CMAKE_SYSTEM_NAME Linux)

set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -std=c++11 -O3 -Ofast -pthread")

include_directories(
	${CMAKE_SOURCE_DIR}/include
)

link_directories(imu_exam
#	${CMAKE_SOURCE_DIR}/libs/rga
)

file(GLOB_RECURSE SRC_FILES
	${CMAKE_SOURCE_DIR}/src/*.cc
)

add_executable(imu_exam
	${SRC_FILES}
)

target_link_libraries(imu_exam
#	drm m rga opencv_core opencv_highgui opencv_imgcodecs opencv_imgproc
)
