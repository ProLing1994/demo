# set core files
file(GLOB CORE_FILES onlinebin/*.cc)

include_directories(${MNN_INCLUDE_DIR})
# add test
add_executable(demo onlinebin/online2-wav-nnet3-latgen-faster.cc)
target_link_libraries(demo ${OPENFST_LIBRARIES} ${KALDI_LIBRARIES})
target_link_libraries(demo ${CUDA_CUDART_LIBRARY} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_cusparse_LIBRARY} ${CUDA_curand_LIBRARY})
target_link_libraries(demo -L/usr/local/cuda-10.0/lib64 -lcusolver)
target_link_libraries(demo -L/opt/intel/mkl/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_sequential)
target_link_libraries(demo -ldl -lpthread -lm)