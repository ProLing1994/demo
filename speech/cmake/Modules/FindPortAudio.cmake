# - Try to find PortAudio
#
# The following variables are optionally searched for defaults
#  PortAudio_DIR:            Base directory where all PortAudio components are found
#
# The following are set after configuration is done:
#  PortAudio_FOUND
#  PortAudio_INCLUDE_DIRS
#  PortAudio_LIBRARIES
#  PortAudio_LIBRARY_DIRS

include(FindPackageHandleStandardArgs)

set(PortAudio_DIR "/home/huanyuan/code/kaldi/kaldi/tools/portaudio" CACHE PATH "Folder contains package PortAudio")

find_path(PortAudio_INCLUDE_DIR portaudio.h
  HINTS ${PortAudio_DIR}
  PATH_SUFFIXES install/include)

find_library(PortAudio_LIBRARY libportaudio.a
  HINTS ${PortAudio_DIR} 
  PATH_SUFFIXES install/lib)

find_package_handle_standard_args(
  PortAudio DEFAULT_MSG PortAudio_INCLUDE_DIR PortAudio_LIBRARY)

if(PortAudio_FOUND)
  set(PortAudio_INCLUDE_DIRS ${PortAudio_INCLUDE_DIR})
  set(PortAudio_LIBRARIES ${PortAudio_LIBRARY})
  include_directories(${PortAudio_INCLUDE_DIRS})
endif()
