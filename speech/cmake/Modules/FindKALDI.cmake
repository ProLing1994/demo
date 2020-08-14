# - Try to find KALDI
#
# The following variables are optionally searched for defaults
#  KALDI_DIR:            Base directory where all KALDI components are found
#
# The following are set after configuration is done:
#  KALDI_FOUND
#  KALDI_INCLUDE_DIRS
#  KALDI_LIBRARIES
#  KALDI_LIBRARY_DIRS

include(FindPackageHandleStandardArgs)

set(KALDI_DIR "/home/huanyuan/code/kaldi/kaldi" CACHE PATH "Folder contains package KALDI")

find_path(KALDI_INCLUDE_DIR feat/wave-reader.h
  HINTS ${KALDI_DIR}
  PATH_SUFFIXES src)

find_library(KALDI_ONLINE2_LIBRARY kaldi-online2.a
  HINTS ${KALDI_DIR} 
  PATH_SUFFIXES src/online2)

find_library(KALDI_IVECTOR_LIBRARY kaldi-ivector.a
  HINTS ${KALDI_DIR} 
  PATH_SUFFIXES src/ivector)

find_library(KALDI_NNET3_LIBRARY kaldi-nnet3.a
  HINTS ${KALDI_DIR} 
  PATH_SUFFIXES src/nnet3)

find_library(KALDI_CHAIN_LIBRARY kaldi-chain.a
  HINTS ${KALDI_DIR} 
  PATH_SUFFIXES src/chain)

find_library(KALDI_CUDAMATRIX_LIBRARY kaldi-cudamatrix.a
  HINTS ${KALDI_DIR} 
  PATH_SUFFIXES src/cudamatrix)

find_library(KALDI_DECODER_LIBRARY kaldi-decoder.a
  HINTS ${KALDI_DIR} 
  PATH_SUFFIXES src/decoder)

find_library(KALDI_LAT_LIBRARY kaldi-lat.a
  HINTS ${KALDI_DIR} 
  PATH_SUFFIXES src/lat)

find_library(KALDI_FSTEXT_LIBRARY kaldi-fstext.a
  HINTS ${KALDI_DIR} 
  PATH_SUFFIXES src/fstext)

find_library(KALDI_HMM_LIBRARY kaldi-hmm.a
  HINTS ${KALDI_DIR} 
  PATH_SUFFIXES src/hmm)

find_library(KALDI_FEAT_LIBRARY kaldi-feat.a
  HINTS ${KALDI_DIR} 
  PATH_SUFFIXES src/feat)

find_library(KALDI_TRANSFORM_LIBRARY kaldi-transform.a
  HINTS ${KALDI_DIR} 
  PATH_SUFFIXES src/transform)

find_library(KALDI_GMM_LIBRARY kaldi-gmm.a
  HINTS ${KALDI_DIR} 
  PATH_SUFFIXES src/gmm)

find_library(KALDI_TREE_LIBRARY kaldi-tree.a
  HINTS ${KALDI_DIR} 
  PATH_SUFFIXES src/tree)

find_library(KALDI_UTIL_LIBRARY kaldi-util.a
  HINTS ${KALDI_DIR} 
  PATH_SUFFIXES src/util)

find_library(KALDI_MATRIX_LIBRARY kaldi-matrix.a
  HINTS ${KALDI_DIR} 
  PATH_SUFFIXES src/matrix)

find_library(KALDI_BASE_LIBRARY kaldi-base.a
  HINTS ${KALDI_DIR} 
  PATH_SUFFIXES src/base)

find_package_handle_standard_args(
  KALDI DEFAULT_MSG KALDI_INCLUDE_DIR KALDI_ONLINE2_LIBRARY KALDI_IVECTOR_LIBRARY KALDI_NNET3_LIBRARY KALDI_CHAIN_LIBRARY 
  KALDI_CUDAMATRIX_LIBRARY KALDI_DECODER_LIBRARY KALDI_LAT_LIBRARY KALDI_FSTEXT_LIBRARY KALDI_HMM_LIBRARY 
  KALDI_FEAT_LIBRARY KALDI_TRANSFORM_LIBRARY KALDI_GMM_LIBRARY KALDI_TREE_LIBRARY KALDI_UTIL_LIBRARY 
  KALDI_MATRIX_LIBRARY KALDI_BASE_LIBRARY)

if(KALDI_FOUND)
  set(KALDI_INCLUDE_DIRS ${KALDI_INCLUDE_DIR})
  set(KALDI_LIBRARIES ${KALDI_ONLINE2_LIBRARY} ${KALDI_IVECTOR_LIBRARY} ${KALDI_NNET3_LIBRARY} ${KALDI_CHAIN_LIBRARY} 
  ${KALDI_CUDAMATRIX_LIBRARY} ${KALDI_DECODER_LIBRARY} ${KALDI_LAT_LIBRARY} ${KALDI_FSTEXT_LIBRARY} ${KALDI_HMM_LIBRARY}
  ${KALDI_FEAT_LIBRARY} ${KALDI_TRANSFORM_LIBRARY} ${KALDI_GMM_LIBRARY} ${KALDI_TREE_LIBRARY} ${KALDI_UTIL_LIBRARY}
  ${KALDI_MATRIX_LIBRARY} ${KALDI_BASE_LIBRARY})
  include_directories(${KALDI_INCLUDE_DIRS})
endif()
