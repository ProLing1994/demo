# set core files
file(GLOB ASR_COMMON_FILES common/*.h common/*.hpp common/*.cpp)
file(GLOB ASR_PYWRAPPERS_FILES pywrappers/*.h pywrappers/*.hpp pywrappers/*.cpp)

source_group(asr_common FILES ${ASR_COMMON_FILES})
source_group(asr_pywrappers FILES ${ASR_PYWRAPPERS_FILES})

# asr dll library
add_library(ai_speech_asr SHARED ${ASR_COMMON_FILES} ${ASR_PYWRAPPERS_FILES})
target_link_libraries(ai_speech_asr ${OpenCV_LIBS})

# add test
add_executable(test_feature test_feature.cpp)
target_link_libraries(test_feature ai_speech_asr ${OpenCV_LIBS})
target_link_libraries(test_feature debug ${COMMON_UTILITY_LIBRARY_DEBUG} optimized ${COMMON_UTILITY_LIBRARY_RELEASE})

# add test
add_executable(test_debug test_debug.cpp)
target_link_libraries(test_debug ai_speech_asr ${OpenCV_LIBS})
target_link_libraries(test_debug debug ${COMMON_UTILITY_LIBRARY_DEBUG} optimized ${COMMON_UTILITY_LIBRARY_RELEASE})

# add test
add_executable(alg_main alg_main.cpp)
target_link_libraries(alg_main ai_speech_asr ${OpenCV_LIBS})
target_link_libraries(alg_main debug ${COMMON_UTILITY_LIBRARY_DEBUG} optimized ${COMMON_UTILITY_LIBRARY_RELEASE})