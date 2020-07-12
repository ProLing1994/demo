# - Try to find MNN
#
# The following variables are optionally searched for defaults
#  MNN_DIR:            Base directory where all MNN components are found
#
# The following are set after configuration is done:
#  MNN_FOUND
#  MNN_INCLUDE_DIRS
#  MNN_LIBRARIES
#  MNN_LIBRARY_DIRS

include(FindPackageHandleStandardArgs)

set(MNN_DIR "/home/huanyuan/code/MNN/build/")

find_path(MNN_INCLUDE_DIR MNN/MNNDefine.h
  HINTS ${MNN_DIR}
  PATH_SUFFIXES install/include)

find_library(MNN_LIBRARY MNN
  HINTS ${MNN_DIR} 
  PATH_SUFFIXES install/lib)

find_package_handle_standard_args(
  MNN DEFAULT_MSG MNN_INCLUDE_DIR MNN_LIBRARY)

if(MNN_FOUND)
  set(MNN_INCLUDE_DIRS ${MNN_INCLUDE_DIR})
  set(MNN_LIBRARIES ${MNN_LIBRARY})
  include_directories(${MNN_INCLUDE_DIR})
endif()
