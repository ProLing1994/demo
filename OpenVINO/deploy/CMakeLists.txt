# set core files
file(GLOB OPENVINO_CORE_FILES inference_detection_openvino.hpp inference_detection_openvino.cpp)
file(GLOB MODEL_CORE_FILES inference_detection_model.hpp inference_detection_model.cpp)
file(GLOB MOSAIC_API_CORE_FILES RMAI_MOSAIC_API.hpp RMAI_MOSAIC_API.cpp)

# yh_mnn dll library
add_library(rmai_mosaic SHARED ${OPENVINO_CORE_FILES} ${MOSAIC_API_CORE_FILES} ${MODEL_CORE_FILES})
target_link_libraries(rmai_mosaic ${OpenCV_LIBS} ${OpenVINO_LIBRARY_DIRS} glog::glog)
target_link_libraries(rmai_mosaic -fvisibility=hidden)

# add test
add_executable(test_mobilenet_ssd test_mobilenet_ssd.cpp)
target_link_libraries(test_mobilenet_ssd rmai_mosaic ${OpenCV_LIBS})
target_link_libraries(test_mobilenet_ssd debug ${YH_COMMON_UTILITY_LIBRARY_DEBUG} optimized ${YH_COMMON_UTILITY_LIBRARY_RELEASE})
target_link_libraries(test_mobilenet_ssd glog::glog)

add_executable(test_mobilenet_ssd_thread test_mobilenet_ssd_thread.cpp)
target_link_libraries(test_mobilenet_ssd_thread rmai_mosaic ${OpenCV_LIBS})
target_link_libraries(test_mobilenet_ssd_thread debug ${YH_COMMON_UTILITY_LIBRARY_DEBUG} optimized ${YH_COMMON_UTILITY_LIBRARY_RELEASE})
target_link_libraries(test_mobilenet_ssd_thread glog::glog)

add_executable(test_mobilenet_ssd_opencv test_mobilenet_ssd_opencv.cpp)
target_link_libraries(test_mobilenet_ssd_opencv ${OpenCV_LIBS})
target_link_libraries(test_mobilenet_ssd_opencv debug ${YH_COMMON_UTILITY_LIBRARY_DEBUG} optimized ${YH_COMMON_UTILITY_LIBRARY_RELEASE})
target_link_libraries(test_mobilenet_ssd_opencv glog::glog)

add_executable(test_mobilenet_ssd_yuv test_mobilenet_ssd_yuv.cpp)
target_link_libraries(test_mobilenet_ssd_yuv rmai_mosaic ${OpenCV_LIBS})
target_link_libraries(test_mobilenet_ssd_yuv glog::glog)

add_executable(test_mosaic_api test_mosaic_api.cpp)
target_link_libraries(test_mosaic_api rmai_mosaic ${OpenCV_LIBS})
target_link_libraries(test_mosaic_api glog::glog)