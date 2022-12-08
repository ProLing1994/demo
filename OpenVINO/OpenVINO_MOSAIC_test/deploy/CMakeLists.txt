# add test
add_executable(test_mosaic_api test_mosaic_api.cpp RMAI_MOSAIC_API.h)
target_link_libraries(test_mosaic_api ${OpenCV_LIBS})
# target_link_libraries(test_mosaic_api optimized E:\\project\\demo\\OpenVINO\\OpenVINO_MOSAIC\\lib\\Release\\rmai_mosaic.lib)
# target_link_libraries(test_mosaic_api debug E:\\project\\demo\\OpenVINO\\OpenVINO_MOSAIC\\lib\\Debug\\rmai_mosaic.lib)
target_link_libraries(test_mosaic_api optimized /home/huanyuan/code/demo/OpenVINO/OpenVINO_MOSAIC/lib/Release/librmai_mosaic.so)
target_link_libraries(test_mosaic_api glog::glog) 