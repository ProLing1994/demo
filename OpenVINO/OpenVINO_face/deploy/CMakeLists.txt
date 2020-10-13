# set core files
file(GLOB FACE_SSD_CORE_FILES face_ssd_caffe_openvino.hpp face_ssd_caffe_openvino.cpp)

# yh_mnn dll library
add_library(rmai_openvino SHARED ${FACE_SSD_CORE_FILES} )
target_link_libraries(rmai_openvino ${OpenVINO_LIBRARY_DIRS} ${OpenCV_LIBS})

# add test
add_executable(test_ssd_face test_ssd_face.cpp)
target_link_libraries(test_ssd_face rmai_openvino ${OpenCV_LIBS})
target_link_libraries(test_ssd_face glog::glog)
