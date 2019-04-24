cmake_minimum_required (VERSION 3.0)

#项目名称
SET(PROJ_NAME "FishEyeCalib")
project (${PROJ_NAME})
set(ThirdParty "${CMAKE_CURRENT_LIST_DIR}/third-party")



SET(INCLUDE_DIRS ${CMAKE_CURRENT_LIST_DIR}/include
                 ${CMAKE_CURRENT_LIST_DIR}/third-party/
				 ${CMAKE_CURRENT_LIST_DIR}/third-party/opencv-3.4/include
				 ${CMAKE_CURRENT_LIST_DIR}/third-party/opencv-3.4/include/opencv
				 ${CMAKE_CURRENT_LIST_DIR}/third-party/opencv-3.4/modules/core/include
				 ${CMAKE_CURRENT_LIST_DIR}/third-party/opencv-3.4/modules/flann/include
				 ${CMAKE_CURRENT_LIST_DIR}/third-party/opencv-3.4/modules/imgproc/include
				 ${CMAKE_CURRENT_LIST_DIR}/third-party/opencv-3.4/modules/ml/include
				 ${CMAKE_CURRENT_LIST_DIR}/third-party/opencv-3.4/modules/objdetect/include
				 ${CMAKE_CURRENT_LIST_DIR}/third-party/opencv-3.4/modules/photo/include
				 ${CMAKE_CURRENT_LIST_DIR}/third-party/opencv-3.4/modules/video/include
				 ${CMAKE_CURRENT_LIST_DIR}/third-party/opencv-3.4/modules/dnn/include
				 ${CMAKE_CURRENT_LIST_DIR}/third-party/opencv-3.4/modules/imgcodecs/include
				 ${CMAKE_CURRENT_LIST_DIR}/third-party/opencv-3.4/modules/shape/include
				 ${CMAKE_CURRENT_LIST_DIR}/third-party/opencv-3.4/modules/videoio/include
				 ${CMAKE_CURRENT_LIST_DIR}/third-party/opencv-3.4/modules/highgui/include
				 ${CMAKE_CURRENT_LIST_DIR}/third-party/opencv-3.4/modules/superres/include
				 ${CMAKE_CURRENT_LIST_DIR}/third-party/opencv-3.4/modules/ts/include
				 ${CMAKE_CURRENT_LIST_DIR}/third-party/opencv-3.4/modules/features2d/include
				 ${CMAKE_CURRENT_LIST_DIR}/third-party/opencv-3.4/modules/calib3d/include
				 ${CMAKE_CURRENT_LIST_DIR}/third-party/opencv-3.4/modules/stitching/include
				 ${CMAKE_CURRENT_LIST_DIR}/third-party/opencv-3.4/modules/videostab/include
				 ${CMAKE_CURRENT_LIST_DIR}/third-party/opencv-3.4/modules/tracking/include
				 ${CMAKE_CURRENT_LIST_DIR}/include/dirent.h
                 )

# 源文件
file(GLOB SOURCE ${CMAKE_CURRENT_LIST_DIR}/src/fisheyecalib.cpp
                 )

# 链接文件
file(GLOB LINK_DIRS ${CMAKE_CURRENT_LIST_DIR}/lib/x64/realsense2.lib
					${CMAKE_CURRENT_LIST_DIR}/lib/x64/glfw-imgui.lib
					${CMAKE_CURRENT_LIST_DIR}/lib/opencv_core341.lib
					${CMAKE_CURRENT_LIST_DIR}/lib/opencv_highgui341.lib
					${CMAKE_CURRENT_LIST_DIR}/lib/opencv_imgproc341.lib
					${CMAKE_CURRENT_LIST_DIR}/lib/opencv_calib3d341.lib
					${CMAKE_CURRENT_LIST_DIR}/lib/opencv_dnn341.lib
					${CMAKE_CURRENT_LIST_DIR}/lib/opencv_features2d341.lib
					${CMAKE_CURRENT_LIST_DIR}/lib/opencv_flann341.lib
					${CMAKE_CURRENT_LIST_DIR}/lib/opencv_imgcodecs341.lib
					${CMAKE_CURRENT_LIST_DIR}/lib/opencv_ml341.lib
					${CMAKE_CURRENT_LIST_DIR}/lib/opencv_objdetect341.lib
					${CMAKE_CURRENT_LIST_DIR}/lib/opencv_photo341.lib
					${CMAKE_CURRENT_LIST_DIR}/lib/opencv_python2.lib
					${CMAKE_CURRENT_LIST_DIR}/lib/opencv_shape341.lib
					${CMAKE_CURRENT_LIST_DIR}/lib/opencv_stitching341.lib
					${CMAKE_CURRENT_LIST_DIR}/lib/opencv_superres341.lib
					${CMAKE_CURRENT_LIST_DIR}/lib/opencv_video341.lib
					${CMAKE_CURRENT_LIST_DIR}/lib/opencv_videoio341.lib
					${CMAKE_CURRENT_LIST_DIR}/lib/opencv_videostab341.lib
					${CMAKE_CURRENT_LIST_DIR}/lib/opencv_tracking341.lib
					${QT_LIB}/x64/Qt5Network.lib
					${QT_LIB}/x64/Qt5Core.lib
					${CMAKE_CURRENT_LIST_DIR}/lib/pthreadVC2.lib
					${CMAKE_CURRENT_LIST_DIR}/lib/libpthreadGC2.a
					)

# 添加头文件
include_directories(${INCLUDE_DIRS})

# 添加源文件
add_executable(${PROJ_NAME} ${SOURCE})

# 编译输出路径
if ( MSVC )
    set_target_properties( ${PROJ_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/bin )
    set_target_properties( ${PROJ_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_LIST_DIR}/bin )
    set_target_properties( ${PROJ_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_LIST_DIR}/bin )
endif ( MSVC )

# 添加链接库
target_link_libraries(${PROJ_NAME} ${LINK_DIRS}
                                   )