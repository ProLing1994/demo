# YH_COMMON C++ library include 
set(YH_COMMON_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR})

# YH_COMMON Link with Libraries
# Make sure that you add ${CMAKE_CURRENT_LIST_DIR}/common/lib/Release 
# into system path.
# On windows, add to environment variable PATH
# On linux, add "export LD_LIBRARY_PATH=xxx/common//lib/Release:$LD_LIBRARY_PATH" to .bashrc file

# YH_COMMON utility library
find_library(YH_COMMON_UTILITY_LIBRARY NAMES yh_common_utility
			HINTS ${CMAKE_CURRENT_LIST_DIR}/common/lib/Release)
find_library(YH_COMMON_UTILITY_LIBRARY_RELEASE NAMES yh_common_utility
			HINTS ${CMAKE_CURRENT_LIST_DIR}/common/lib/Release)
find_library(YH_COMMON_UTILITY_LIBRARY_DEBUG NAMES yh_common_utility
			HINTS ${CMAKE_CURRENT_LIST_DIR}/common/lib/Debug)  
                      
# MD libraries
set(YH_COMMON_LIBRARIES ${YH_COMMON_UTILITY_LIBRARY})
set(YH_COMMON_LIBRARIES_RELEASE ${YH_COMMON_UTILITY_LIBRARY_RELEASE})
set(YH_COMMON_LIBRARIES_DEBUG ${YH_COMMON_UTILITY_LIBRARY_DEBUG})

# MD utility library
set(YH_COMMON_UTILITY_ROOT ${CMAKE_CURRENT_LIST_DIR}/common/utils/csrc)
file(GLOB YH_COMMON_UTILITY_SRC ${YH_COMMON_UTILITY_ROOT}/*.h ${YH_COMMON_UTILITY_ROOT}/*.cpp)