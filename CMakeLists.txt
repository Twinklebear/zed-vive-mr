cmake_minimum_required(VERSION 3.5)
project(ZED_MR)
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${ZED_MR_SOURCE_DIR}/cmake")

add_definitions(-DGLM_FORCE_RADIANS)
find_package(OpenGL REQUIRED)
find_package(SDL2 REQUIRED)
find_package(GLM REQUIRED)
find_package(OpenVRSDK REQUIRED)
find_package(ZED 2 REQUIRED)
find_package(CUDA REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS} ${ZED_INCLUDE_DIRS}
	${SDL2_INCLUDE_DIR} ${OPENGL_INCLUDE_DIR} ${GLM_INCLUDE_DIRS}
	${OPENVR_SDK_INCLUDE_DIR})

add_executable(zed_test main.cpp gl_core_4_5.c debug.cpp
	openvr_display.cpp util.cpp obj_model.cpp)
set_property(TARGET zed_test PROPERTY CXX_STANDARD 14)
target_link_libraries(zed_test ${OPENGL_LIBRARIES} ${SDL2_LIBRARY}
	${ZED_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY}
	${CUDA_NPP_LIBRARIES_ZED} ${OPENVR_SDK_LIBRARY})
	

