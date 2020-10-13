# this one is important
SET(CMAKE_SYSTEM_NAME Linux)
 
# specify the cross compiler
SET(CMAKE_C_COMPILER  arm-himix200-linux-gcc)
SET(CMAKE_CXX_COMPILER arm-himix200-linux-g++)
 
# where is the target environment 
SET(CMAKE_FIND_ROOT_PATH  /home/workspace/RMAI/third_party/)
 
# search for programs in the build host directories
SET(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
# for libraries and headers in the target directories
SET(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
SET(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)