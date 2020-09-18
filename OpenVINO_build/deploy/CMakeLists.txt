# add test
add_executable(test_mosaic_api test_mosaic_api.cpp RMAI_MOSAIC_API.h)
target_link_libraries(test_mosaic_api F:\\software\\demo\\OpenVINO\\lib\\Release\\rmai_mosaic.lib ${OpenCV_LIBS})
target_link_libraries(test_mosaic_api glog::glog) 