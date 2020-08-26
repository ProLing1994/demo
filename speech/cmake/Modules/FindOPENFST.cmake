# - Try to find OPENFST
#
# The following variables are optionally searched for defaults
#  OPENFST_DIR:            Base directory where all OPENFST components are found
#
# The following are set after configuration is done:
#  OPENFST_FOUND
#  OPENFST_INCLUDE_DIRS
#  OPENFST_LIBRARIES
#  OPENFST_LIBRARY_DIRS

include(FindPackageHandleStandardArgs)

set(OPENFST_DIR "/home/huanyuan/code/kaldi/kaldi/tools/openfst-1.6.7" CACHE PATH "Folder contains package OPENFST")

find_path(OPENFST_INCLUDE_DIR fst/types.h
  HINTS ${OPENFST_DIR}
  PATH_SUFFIXES include)

find_library(OPENFST_LIBRARY fst
  HINTS ${OPENFST_DIR} 
  PATH_SUFFIXES lib)

find_package_handle_standard_args(
  OPENFST DEFAULT_MSG OPENFST_INCLUDE_DIR OPENFST_LIBRARY)

if(OPENFST_FOUND)
  set(OPENFST_INCLUDE_DIRS ${OPENFST_INCLUDE_DIR})
  set(OPENFST_LIBRARIES ${OPENFST_LIBRARY})
  include_directories(${OPENFST_INCLUDE_DIRS})
endif()
