# set core files
file(GLOB OPENVINO_CORE_FILES mobilenet_ssd_detector.h mobilenet_ssd_detector.cpp)

# yh_mnn dll library
add_library(yh_openvino SHARED ${OPENVINO_CORE_FILES})
target_link_libraries(yh_openvino ${OpenCV_LIBS} ${OpenVINO_LIBRARY_DIRS} glog::glog)

# add test
add_executable(test_mobilenet_ssd test_mobilenet_ssd.cpp)
target_link_libraries(test_mobilenet_ssd yh_openvino ${OpenCV_LIBS})
target_link_libraries(test_mobilenet_ssd debug ${YH_COMMON_UTILITY_LIBRARY_DEBUG} optimized ${YH_COMMON_UTILITY_LIBRARY_RELEASE})
target_link_libraries(test_mobilenet_ssd glog::glog)

add_executable(test_mobilenet_ssd_thread test_mobilenet_ssd_thread.cpp)
target_link_libraries(test_mobilenet_ssd_thread yh_openvino ${OpenCV_LIBS})
target_link_libraries(test_mobilenet_ssd_thread debug ${YH_COMMON_UTILITY_LIBRARY_DEBUG} optimized ${YH_COMMON_UTILITY_LIBRARY_RELEASE})
target_link_libraries(test_mobilenet_ssd_thread glog::glog)

add_executable(test_mobilenet_ssd_opencv test_mobilenet_ssd_opencv.cpp)
target_link_libraries(test_mobilenet_ssd_opencv ${OpenCV_LIBS})
target_link_libraries(test_mobilenet_ssd_opencv debug ${YH_COMMON_UTILITY_LIBRARY_DEBUG} optimized ${YH_COMMON_UTILITY_LIBRARY_RELEASE})
target_link_libraries(test_mobilenet_ssd_opencv glog::glog)

add_executable(test_mobilenet_ssd_yuv test_mobilenet_ssd_yuv.cpp)
target_link_libraries(test_mobilenet_ssd_yuv yh_openvino ${OpenCV_LIBS})
target_link_libraries(test_mobilenet_ssd_yuv debug ${YH_COMMON_UTILITY_LIBRARY_DEBUG} optimized ${YH_COMMON_UTILITY_LIBRARY_RELEASE})
target_link_libraries(test_mobilenet_ssd_yuv glog::glog)