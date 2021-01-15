# set core files
file(GLOB ASR_COMMON_FILES common/*.h common/*.hpp common/*.cpp )

source_group(asr_common FILES ${ASR_COMMON_FILES})

# asr dll library
add_library(ai_speech_asr SHARED ${ASR_COMMON_FILES})
target_link_libraries(ai_speech_asr ${OpenCV_LIBS} )
target_link_libraries(ai_speech_asr glog::glog)

# add test
add_executable(test_asr_demo test_asr_demo.cpp)
target_link_libraries(test_asr_demo ai_speech_asr ${OpenCV_LIBS})
target_link_libraries(test_asr_demo debug ${COMMON_UTILITY_LIBRARY_DEBUG} optimized ${COMMON_UTILITY_LIBRARY_RELEASE})
target_link_libraries(test_asr_demo glog::glog)