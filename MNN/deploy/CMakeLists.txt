# set core files
file(GLOB MNN_CORE_FILES mobilenet_ssd_detector.h mobilenet_ssd_detector.cpp)

# yh_mnn dll library
add_library(yh_mnn SHARED ${MNN_CORE_FILES})
target_link_libraries(yh_mnn ${OpenCV_LIBS} ${MNN_LIBRARY} glog::glog)

# add test
add_executable(test_mobilenet_ssd test_mobilenet_ssd.cpp)
target_link_libraries(test_mobilenet_ssd yh_mnn ${OpenCV_LIBS})
target_link_libraries(test_mobilenet_ssd debug ${YH_COMMON_UTILITY_LIBRARY_DEBUG} optimized ${YH_COMMON_UTILITY_LIBRARY_RELEASE})
target_link_libraries(test_mobilenet_ssd glog::glog)

add_executable(test_mobilenet_ssd_thread test_mobilenet_ssd_thread.cpp)
target_link_libraries(test_mobilenet_ssd_thread yh_mnn ${OpenCV_LIBS})
target_link_libraries(test_mobilenet_ssd_thread debug ${YH_COMMON_UTILITY_LIBRARY_DEBUG} optimized ${YH_COMMON_UTILITY_LIBRARY_RELEASE})
target_link_libraries(test_mobilenet_ssd_thread glog::glog)