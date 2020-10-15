# COMMON C++ library include 
set(COMMON_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR})

# COMMON Link with Libraries
# Make sure that you add ${CMAKE_CURRENT_LIST_DIR}/common/lib/Release 
# into system path.
# On windows, add to environment variable PATH
# On linux, add "export LD_LIBRARY_PATH=xxx/common//lib/Release:$LD_LIBRARY_PATH" to .bashrc file

# COMMON utility library
find_library(COMMON_UTILITY_LIBRARY NAMES common_utility
			HINTS ${CMAKE_CURRENT_LIST_DIR}/common/lib/Release)
find_library(COMMON_UTILITY_LIBRARY_RELEASE NAMES common_utility
			HINTS ${CMAKE_CURRENT_LIST_DIR}/common/lib/Release)
find_library(COMMON_UTILITY_LIBRARY_DEBUG NAMES common_utility
			HINTS ${CMAKE_CURRENT_LIST_DIR}/common/lib/Debug)  
                      
# MD libraries
set(COMMON_LIBRARIES ${COMMON_UTILITY_LIBRARY})
set(COMMON_LIBRARIES_RELEASE ${COMMON_UTILITY_LIBRARY_RELEASE})
set(COMMON_LIBRARIES_DEBUG ${COMMON_UTILITY_LIBRARY_DEBUG})

# MD utility library
set(COMMON_UTILITY_ROOT ${CMAKE_CURRENT_LIST_DIR}/common/utils/csrc)
file(GLOB COMMON_UTILITY_SRC ${COMMON_UTILITY_ROOT}/*.h ${COMMON_UTILITY_ROOT}/*.cpp)