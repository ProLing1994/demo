# set core files
file(GLOB ASR_COMMON_FILES common/*.h common/*.hpp common/*.cpp)

source_group(asr_common FILES ${ASR_COMMON_FILES})

# asr dll library
add_library(ai_speech_asr SHARED ${ASR_COMMON_FILES} )
target_link_libraries(ai_speech_asr ${OpenCV_LIBS})
target_link_libraries(ai_speech_asr fftw3 fftw3f)

# 宏
add_definitions(-D_TESTTIME)
add_definitions(-D_DEBUG)

# add test
add_executable(alg_main alg_main.cpp)
target_link_libraries(alg_main ai_speech_asr)
target_link_libraries(alg_main ${OpenCV_LIBS})
target_link_libraries(alg_main fftw3 fftw3f)
target_link_libraries(alg_main debug ${COMMON_UTILITY_LIBRARY_DEBUG} optimized ${COMMON_UTILITY_LIBRARY_RELEASE})