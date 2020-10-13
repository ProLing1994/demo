USE_GCC=n

PROJECT_DIR=$(shell pwd)
THIRD_PARTY_DIR:=/home/workspace/RMAI/third_party

ifeq (${PRODUCT_TYPE}, ADkit_3516DV300)
	CFLAGS += -mcpu=cortex-a53 -mfloat-abi=softfp -mfpu=neon-vfpv4 -mno-unaligned-access -fno-aggressive-loop-optimizations -W -Wall -O3 -std=c++11 -DNNIE
	CXXFlAGS += -mcpu=cortex-a53 -mfloat-abi=softfp -mfpu=neon-vfpv4 -mno-unaligned-access -fno-aggressive-loop-optimizations -W -Wall -O3 -std=c++11 -DNNIE
else ifeq (${PRODUCT_TYPE}, ADkit_3519AV100)
	CFLAGS += -mcpu=cortex-a53 -mfloat-abi=softfp -mfpu=neon-vfpv4 -mno-unaligned-access -fno-aggressive-loop-optimizations -W -Wall -O3 -std=c++11 -DNNIE
	CXXFlAGS += -mcpu=cortex-a53 -mfloat-abi=softfp -mfpu=neon-vfpv4 -mno-unaligned-access -fno-aggressive-loop-optimizations -W -Wall -O3 -std=c++11 -DNNIE
endif

LDFLAGS += -lpthread -lm -ldl

ifeq (${PRODUCT_TYPE}, ADkit_3516DV300)
	# INCLUDE_DIR+=-I${THIRD_PARTY_DIR}/opencv2411/include/opencv
	# INCLUDE_DIR+=-I${THIRD_PARTY_DIR}/opencv2411/include/opencv2
	# INCLUDE_DIR+=-I${THIRD_PARTY_DIR}/opencv2411/include
	# INCLUDE_DIR+=-I${THIRD_PARTY_DIR}/hisi_sdk_3516DV300/include
	# INCLUDE_DIR+=-I${THIRD_PARTY_DIR}/rmai_lib_3516DV300/include
	INCLUDE_DIR+=-I${PROJECT_DIR}/demo
else ifeq (${PRODUCT_TYPE}, ADkit_3519AV100)
	# INCLUDE_DIR+=-I${THIRD_PARTY_DIR}/opencv2411/include/opencv
	# INCLUDE_DIR+=-I${THIRD_PARTY_DIR}/opencv2411/include/opencv2
	# INCLUDE_DIR+=-I${THIRD_PARTY_DIR}/opencv2411/include
	# INCLUDE_DIR+=-I${THIRD_PARTY_DIR}/hisi_sdk_3519AV100/include
	# INCLUDE_DIR+=-I${THIRD_PARTY_DIR}/rmai_lib_3519AV100/include
	INCLUDE_DIR+=-I${PROJECT_DIR}/demo
endif

ifeq (${PRODUCT_TYPE}, ADkit_3516DV300)
	SRC_DIR+=${PROJECT_DIR}/demo
else ifeq (${PRODUCT_TYPE}, ADkit_3519AV100)
	SRC_DIR+=${PROJECT_DIR}/demo	
endif

SRC_FILE_C+=
SRC_FILE_CPP+=

ifeq (${PRODUCT_TYPE}, ADkit_3516DV300)
	# OPENCV_LIB_DIR:=${THIRD_PARTY_DIR}/opencv2411/lib_himix200
	# HISI_SDK_LIB_DIR:=${THIRD_PARTY_DIR}/hisi_sdk_3516DV300/lib
	# RMAI_LIB_DIR:=${THIRD_PARTY_DIR}/rmai_lib_3516DV300/lib
	# LIB_DIR+=-L${OPENCV_LIB_DIR}
	# LIB_DIR+=-L${HISI_SDK_LIB_DIR}
	# LIB_DIR+=-L${RMAI_LIB_DIR}
	LIB_DIR+=-L${PROJECT_DIR}/demo/
else ifeq (${PRODUCT_TYPE}, ADkit_3519AV100)
	# OPENCV_LIB_DIR:=${THIRD_PARTY_DIR}/opencv2411/lib_himix200
	# HISI_SDK_LIB_DIR:=${THIRD_PARTY_DIR}/hisi_sdk_3519AV100/lib
	# RMAI_LIB_DIR:=${THIRD_PARTY_DIR}/rmai_lib_3519AV100/lib
	# LIB_DIR+=-L${OPENCV_LIB_DIR}
	# LIB_DIR+=-L${HISI_SDK_LIB_DIR}
	# LIB_DIR+=-L${RMAI_LIB_DIR}
	LIB_DIR+=-L${PROJECT_DIR}/demo/
endif

ifeq (${PRODUCT_TYPE}, ADkit_3516DV300)
	# STATIC_LIB+=${OPENCV_LIB_DIR}/libopencv_objdetect.a
	# STATIC_LIB+=${OPENCV_LIB_DIR}/libopencv_highgui.a
	# STATIC_LIB+=${OPENCV_LIB_DIR}/libopencv_video.a
	# STATIC_LIB+=${OPENCV_LIB_DIR}/libopencv_imgproc.a
	# STATIC_LIB+=${OPENCV_LIB_DIR}/libopencv_core.a
	# STATIC_LIB+=${OPENCV_LIB_DIR}/libzlib.a
	# STATIC_LIB+=${OPENCV_LIB_DIR}/liblibjpeg.a
	# STATIC_LIB+=${OPENCV_LIB_DIR}/liblibpng.a
	# STATIC_LIB+=${OPENCV_LIB_DIR}/liblibjasper.a

	# STATIC_LIB+=${HISI_SDK_LIB_DIR}/libmpi.a
	# STATIC_LIB+=${HISI_SDK_LIB_DIR}/libsecurec.a
	# STATIC_LIB+=${HISI_SDK_LIB_DIR}/libive.a
	# STATIC_LIB+=${HISI_SDK_LIB_DIR}/libmd.a
	# STATIC_LIB+=${HISI_SDK_LIB_DIR}/libnnie.a
	# STATIC_LIB+=${HISI_SDK_LIB_DIR}/libVoiceEngine.a
	# STATIC_LIB+=${HISI_SDK_LIB_DIR}/libupvqe.a
	# STATIC_LIB+=${HISI_SDK_LIB_DIR}/libdnvqe.a
	# STATIC_LIB+=${HISI_SDK_LIB_DIR}/lib_hiae.a
	# STATIC_LIB+=${HISI_SDK_LIB_DIR}/libisp.a
	# STATIC_LIB+=${HISI_SDK_LIB_DIR}/lib_hidehaze.a
	# STATIC_LIB+=${HISI_SDK_LIB_DIR}/lib_hidrc.a
	# STATIC_LIB+=${HISI_SDK_LIB_DIR}/lib_hildci.a
	# STATIC_LIB+=${HISI_SDK_LIB_DIR}/lib_hiawb.a
	# STATIC_LIB+=${HISI_SDK_LIB_DIR}/libsns_imx327.a
	# STATIC_LIB+=${HISI_SDK_LIB_DIR}/libsns_imx327_2l.a
	# STATIC_LIB+=${HISI_SDK_LIB_DIR}/libsns_imx307.a
	# STATIC_LIB+=${HISI_SDK_LIB_DIR}/libsns_imx335.a
	# STATIC_LIB+=${HISI_SDK_LIB_DIR}/libsns_imx458.a
	# STATIC_LIB+=${HISI_SDK_LIB_DIR}/libsns_mn34220.a
else ifeq (${PRODUCT_TYPE}, ADkit_3519AV100)
	# STATIC_LIB+=${OPENCV_LIB_DIR}/libopencv_objdetect.a
	# STATIC_LIB+=${OPENCV_LIB_DIR}/libopencv_highgui.a
	# STATIC_LIB+=${OPENCV_LIB_DIR}/libopencv_video.a
	# STATIC_LIB+=${OPENCV_LIB_DIR}/libopencv_imgproc.a
	# STATIC_LIB+=${OPENCV_LIB_DIR}/libopencv_core.a
	# STATIC_LIB+=${OPENCV_LIB_DIR}/libzlib.a
	# STATIC_LIB+=${OPENCV_LIB_DIR}/liblibjpeg.a
	# STATIC_LIB+=${OPENCV_LIB_DIR}/liblibpng.a
	# STATIC_LIB+=${OPENCV_LIB_DIR}/liblibjasper.a

	# STATIC_LIB+=${HISI_SDK_LIB_DIR}/libmpi.a
	# STATIC_LIB+=${HISI_SDK_LIB_DIR}/libsecurec.a
	# STATIC_LIB+=${HISI_SDK_LIB_DIR}/libive.a
	# STATIC_LIB+=${HISI_SDK_LIB_DIR}/libmd.a
	# STATIC_LIB+=${HISI_SDK_LIB_DIR}/libnnie.a
	# STATIC_LIB+=${HISI_SDK_LIB_DIR}/libVoiceEngine.a
	# STATIC_LIB+=${HISI_SDK_LIB_DIR}/libupvqe.a
	# STATIC_LIB+=${HISI_SDK_LIB_DIR}/libdnvqe.a
	# STATIC_LIB+=${HISI_SDK_LIB_DIR}/lib_hiae.a
	# STATIC_LIB+=${HISI_SDK_LIB_DIR}/libisp.a
	# STATIC_LIB+=${HISI_SDK_LIB_DIR}/lib_hidehaze.a
	# STATIC_LIB+=${HISI_SDK_LIB_DIR}/lib_hidrc.a
	# STATIC_LIB+=${HISI_SDK_LIB_DIR}/lib_hildci.a
	# STATIC_LIB+=${HISI_SDK_LIB_DIR}/lib_hiawb.a
	# STATIC_LIB+=${HISI_SDK_LIB_DIR}/libsns_imx290.a
	# STATIC_LIB+=${HISI_SDK_LIB_DIR}/libsns_imx290_slave.a
	# STATIC_LIB+=${HISI_SDK_LIB_DIR}/libsns_imx334.a
endif	

ifeq (${PRODUCT_TYPE}, ADkit_3516DV300)
	# DYN_LIB+=-lrmai_nnie
	# DYN_LIB+=-lrmai_kws
else ifeq (${PRODUCT_TYPE}, ADkit_3519AV100)
	# DYN_LIB+=-lrmai_nnie
endif

AIM_NAME=demo_hello_world
#AIM_NAME=XXX
#AIM_NAME=libXXX.so
#AIM_NAME=libXXX.a
#AIM_NAME_EX=XXX
#AIM_NAME_EX=libXXX.so
#AIM_NAME_EX=libXXX.a

ifdef RELEASE_DIR
define RELEASE_INCLUDE_FILE
	@mkdir -p ${RELEASE_DIR}/include
	@echo "PROJECT_DIR", ${PROJECT_DIR}
	@cp ${PROJECT_DIR}/src/XXX.hpp ${RELEASE_DIR}/include
endef
endif

BIN_OUTPUT_DIR=/home/workspace/RMAI/bin_output/yuanhuan/HelloWorld/