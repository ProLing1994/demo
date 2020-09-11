# set core files
file(GLOB OPENVINO_CORE_FILES inference_detection_openvino.h inference_detection_openvino.cpp)

# yh_mnn dll library
add_library(librmai_mosaic SHARED ${OPENVINO_CORE_FILES})
target_link_libraries(librmai_mosaic ${OpenCV_LIBS} ${OpenVINO_LIBRARY_DIRS} glog::glog)
target_link_libraries(librmai_mosaic -fvisibility=hidden)

# add test
add_executable(test_mobilenet_ssd test_mobilenet_ssd.cpp)
target_link_libraries(test_mobilenet_ssd librmai_mosaic ${OpenCV_LIBS})
target_link_libraries(test_mobilenet_ssd debug ${YH_COMMON_UTILITY_LIBRARY_DEBUG} optimized ${YH_COMMON_UTILITY_LIBRARY_RELEASE})
target_link_libraries(test_mobilenet_ssd glog::glog)

add_executable(test_mobilenet_ssd_thread test_mobilenet_ssd_thread.cpp)
target_link_libraries(test_mobilenet_ssd_thread librmai_mosaic ${OpenCV_LIBS})
target_link_libraries(test_mobilenet_ssd_thread debug ${YH_COMMON_UTILITY_LIBRARY_DEBUG} optimized ${YH_COMMON_UTILITY_LIBRARY_RELEASE})
target_link_libraries(test_mobilenet_ssd_thread glog::glog)

add_executable(test_mobilenet_ssd_opencv test_mobilenet_ssd_opencv.cpp)
target_link_libraries(test_mobilenet_ssd_opencv ${OpenCV_LIBS})
target_link_libraries(test_mobilenet_ssd_opencv debug ${YH_COMMON_UTILITY_LIBRARY_DEBUG} optimized ${YH_COMMON_UTILITY_LIBRARY_RELEASE})
target_link_libraries(test_mobilenet_ssd_opencv glog::glog)

add_executable(test_mobilenet_ssd_yuv test_mobilenet_ssd_yuv.cpp)
target_link_libraries(test_mobilenet_ssd_yuv librmai_mosaic ${OpenCV_LIBS})
target_link_libraries(test_mobilenet_ssd_yuv debug ${YH_COMMON_UTILITY_LIBRARY_DEBUG} optimized ${YH_COMMON_UTILITY_LIBRARY_RELEASE})
target_link_libraries(test_mobilenet_ssd_yuv glog::glog)