# set core files
file(GLOB CORE_FILES onlinebin/*.cc)

include_directories(${MNN_INCLUDE_DIR})

# add test
add_executable(online2-wav-nnet3-latgen-faster onlinebin/online2-wav-nnet3-latgen-faster.cc)
target_link_libraries(online2-wav-nnet3-latgen-faster ${OPENFST_LIBRARIES} ${KALDI_LIBRARIES})
target_link_libraries(online2-wav-nnet3-latgen-faster ${CUDA_CUDART_LIBRARY} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_cusparse_LIBRARY} ${CUDA_curand_LIBRARY})
target_link_libraries(online2-wav-nnet3-latgen-faster -L/usr/local/cuda-10.0/lib64 -lcusolver)
target_link_libraries(online2-wav-nnet3-latgen-faster -L/opt/intel/mkl/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_sequential)
target_link_libraries(online2-wav-nnet3-latgen-faster -ldl -lpthread -lm)

# add test
add_executable(online2-tcp-nnet3-decode-faster onlinebin/online2-tcp-nnet3-decode-faster.cc)
target_link_libraries(online2-tcp-nnet3-decode-faster ${OPENFST_LIBRARIES} ${KALDI_LIBRARIES})
target_link_libraries(online2-tcp-nnet3-decode-faster ${CUDA_CUDART_LIBRARY} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_cusparse_LIBRARY} ${CUDA_curand_LIBRARY})
target_link_libraries(online2-tcp-nnet3-decode-faster -L/usr/local/cuda-10.0/lib64 -lcusolver)
target_link_libraries(online2-tcp-nnet3-decode-faster -L/opt/intel/mkl/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_sequential)
target_link_libraries(online2-tcp-nnet3-decode-faster -ldl -lpthread -lm)

# add test
add_executable(online2-simulation-nnet3-decode-faster onlinebin/online2-simulation-nnet3-decode-faster.cc)
target_link_libraries(online2-simulation-nnet3-decode-faster ${OPENFST_LIBRARIES} ${KALDI_LIBRARIES} ${PortAudio_LIBRARIES})
target_link_libraries(online2-simulation-nnet3-decode-faster ${CUDA_CUDART_LIBRARY} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_cusparse_LIBRARY} ${CUDA_curand_LIBRARY})
target_link_libraries(online2-simulation-nnet3-decode-faster -L/usr/local/cuda-10.0/lib64 -lcusolver)
target_link_libraries(online2-simulation-nnet3-decode-faster -L/opt/intel/mkl/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_sequential)
target_link_libraries(online2-simulation-nnet3-decode-faster -ldl -lpthread -lm -lasound -lrt)

# add test
add_executable(online2-nnet3-decode-faster onlinebin/online2-nnet3-decode-faster.cc)
target_link_libraries(online2-nnet3-decode-faster ${OPENFST_LIBRARIES} ${KALDI_LIBRARIES} ${PortAudio_LIBRARIES})
target_link_libraries(online2-nnet3-decode-faster ${CUDA_CUDART_LIBRARY} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_cusparse_LIBRARY} ${CUDA_curand_LIBRARY})
target_link_libraries(online2-nnet3-decode-faster -L/usr/local/cuda-10.0/lib64 -lcusolver)
target_link_libraries(online2-nnet3-decode-faster -L/opt/intel/mkl/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_sequential)
target_link_libraries(online2-nnet3-decode-faster -ldl -lpthread -lm -lasound -lrt)

# add test
add_executable(online-wav-gmm-decode-faster onlinebin/online-wav-gmm-decode-faster.cc)
target_link_libraries(online-wav-gmm-decode-faster ${OPENFST_LIBRARIES} ${KALDI_LIBRARIES} ${PortAudio_LIBRARIES})
target_link_libraries(online-wav-gmm-decode-faster ${CUDA_CUDART_LIBRARY} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_cusparse_LIBRARY} ${CUDA_curand_LIBRARY})
target_link_libraries(online-wav-gmm-decode-faster -L/usr/local/cuda-10.0/lib64 -lcusolver)
target_link_libraries(online-wav-gmm-decode-faster -L/opt/intel/mkl/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_sequential)
target_link_libraries(online-wav-gmm-decode-faster -ldl -lpthread -lm -lasound -lrt)

# add test
add_executable(online-gmm-decode-faster onlinebin/online-gmm-decode-faster.cc)
target_link_libraries(online-gmm-decode-faster ${OPENFST_LIBRARIES} ${KALDI_LIBRARIES} ${PortAudio_LIBRARIES})
target_link_libraries(online-gmm-decode-faster ${CUDA_CUDART_LIBRARY} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_cusparse_LIBRARY} ${CUDA_curand_LIBRARY})
target_link_libraries(online-gmm-decode-faster -L/usr/local/cuda-10.0/lib64 -lcusolver)
target_link_libraries(online-gmm-decode-faster -L/opt/intel/mkl/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_sequential)
target_link_libraries(online-gmm-decode-faster -ldl -lpthread -lm -lasound -lrt)