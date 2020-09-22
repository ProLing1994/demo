#--需指定配置路径--
# TENSORRT_PATH = /media/user/D6DA8490DA846F15/下载/cuda-all-drive/TensorRT-7.0.0.11
OPENVINO_PATH = ./demo/openvino
# JDK_PATH := /home/easyits/product/jdk
# CUDAS_PATH := /usr/local/cuda-10.0
# FAISS_PATH :=#若不启用search功能，该部分保留
#---------------

#----版本信息----
# CUDA_VER = 10.0
# CUDNN_VER = 7.6
# GPU_VER = 1080
# TRT_VER = 7.0.0.11
SYS_VER = U16.04
#--------------

#----算法库信息----
LIB_VER = ver4.0_CPU
MODEL_VER = 20200821
LIB_NAME:=libfaceRec.$(LIB_VER).so
DEMO_NAME:=test_ssd_face
#----------------

#----用于版本验证的算法库与模型库信息----
# INLINE_LIBVERSION:=FaceServer-$(LIB_VER)-$(SYS_VER)-$(CUDA_VER)-$(CUDNN_VER)-$(TRT_VER)-$(GPU_VER)
INLINE_LIBVERSION:=FaceServer-$(LIB_VER)-$(SYS_VER)
INLINE_MODELVERSION:=Mode-$(MODEL_VER)
#----------------

#----功能选择----
MODE = 0#0--生成本地DEBUG测试程序，1--生成RELEASE版本库
DEBUG = 0#0--关闭调试打印信息。当MODE=1时，强制DEBUG=0
SEARCH_APP = 0#search功能，0--不启用，1--启用
MASK_APP = 0#口罩识别功能，0--不启用，1--只检测司机
#---------------

#----通配符----
CC = g++
NVCC = $(CUDAS_PATH)/bin/nvcc
# DLSW_TRIPLE ?= x86_64-linux-gnu
COMMON_LIBS += -lstdc++
# COMMON_LIBS = -lcudnn -lcublas -lcudart
# COMMON_LIBS += -lopencv_calib3d -lopencv_features2d -lopencv_video -lopencv_ml -lopencv_highgui -lopencv_imgproc -lopencv_core
# COMMON_LIBS += -lopencv_contrib -lopencv_legacy -lopencv_flann -lopencv_objdetect
COMMON_LIBS += -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lopencv_core
COMMON_LIBS += -lrt -ldl -lpthread
# tensorrt相关lib
# COMMON_LIBS += -lmyelin -lnvrtc
COMMON_LIBS += -linference_engine -linference_engine_legacy

# LIBS = -lnvinfer -lnvparsers -lnvinfer_plugin -lnvonnxparser $(COMMON_LIBS)
LIBS = $(COMMON_LIBS)


# CFLAGS
GCC_FLAGS += -O3 -fPIC -Wall -Wno-deprecated-declarations -std=c++11
# CUDA_FLAGS += -std=c++11 -Xcompiler -fpic

#-------------


#----通配路径----
PRJ_ROOT := $(shell pwd)
OBJECTS_DIR:=$(PRJ_ROOT)/build
SRC_DIR := $(PRJ_ROOT)/deploy

#--源码路径
# PROJECT_SRC+=$(wildcard $(PRJ_ROOT)/src/plugins/*.cpp)# plugins通用插件
# PROJECT_SRC+=$(wildcard $(PRJ_ROOT)/src/plugins/PReLUPlugin/*.cu)# PReLU插件
# PROJECT_SRC+=$(wildcard $(PRJ_ROOT)/src/caffetrt/face*.cpp)# Caffe推理网络，人脸检测、对齐、识别部分
# PROJECT_SRC+=$(wildcard $(PRJ_ROOT)/src/apis/*.cpp)# FaceSearchAPI
# PROJECT_SRC+=$(wildcard $(PRJ_ROOT)/src/*.cpp)# main函数
# PROJECT_SRC+=$(wildcard $(PRJ_ROOT)/deploy/*.cpp)# openvino函数
PROJECT_SRC+=$(PRJ_ROOT)/deploy/face_ssd_caffe_openvino.cpp# openvino函数
PROJECT_SRC+=$(PRJ_ROOT)/deploy/facerec_caffe_openvino.cpp# openvino函数
PROJECT_SRC+=$(PRJ_ROOT)/deploy/test_ssd_face_wo_glog.cpp# openvino函数
#--------

#--头文件路径
# INCPATHS += -I"/usr/include/opencv2"
INCPATHS += -I./demo/opencv/include
# INCPATHS += -I"$(CUDAS_PATH)/include"
# INCPATHS += -I"/usr/include/$(DLSW_TRIPLE)"
INCPATHS += -I"$(SRC_DIR)"
INCPATHS += -I"$(OPENVINO_PATH)/include"
# INCPATHS += -I"$(OPENVINO_PATH)/deployment_tools/inference_engine/include"
# INCPATHS += -I"$(SRC_DIR)/tensorrt7"
# INCPATHS += -I"$(SRC_DIR)/common"
# INCPATHS += -I"$(SRC_DIR)/plugins"
# INCPATHS += -I"$(SRC_DIR)/plugins/PReLUPlugin"
#--------

#--库路径
LIBPATHS += -L./demo/opencv/libs
# LIBPATHS += -L"/usr/lib/$(DLSW_TRIPLE)/stubs" -L"/usr/lib/$(DLSW_TRIPLE)"
LIBPATHS += -L"$(OPENVINO_PATH)/libs/intel64"
# LIBPATHS += -L"$(OPENVINO_PATH)/deployment_tools/inference_engine/lib/intel64"
# LIBPATHS += -L"$(CUDAS_PATH)/lib64"
#--------
#--------------

#----MODE编译模式----
ifeq ($(MODE),1)
	DEBUG=0
	PROJECT_SRC+=$(wildcard $(PRJ_ROOT)/src/FaceRecAPIs.cpp)# API接口
	INCPATHS+=-I"$(JDK_PATH)/include"
	INCPATHS+=-I"$(JDK_PATH)/include/linux"
	LIBPATHS+=-I"$(JDK_PATH)/lib/amd64"
else
	PROJECT_SRC+=$(wildcard $(PRJ_ROOT)/src/main.cpp)# main函数
endif
DEFS += -DDEBUG=$(DEBUG)
DEFS += -DINLINE_LIBVERSION=\"$(INLINE_LIBVERSION)\"
#--------------

#----SEARCH功能----
DEFS += -DSEARCH_APP=$(SEARCH_APP)
ifeq ($(SEARCH_APP),1)
	INCPATHS+=-I"$(FAISS_PATH)"
	LIBPATHS+=-L"$(FAISS_PATH)"
	LIBS+=-lfaiss
endif
#--------------

#----MASK功能----
DEFS += -DMASK_APP=$(MASK_APP)
ifneq ($(MASK_APP),0)
	PROJECT_SRC+=$(PRJ_ROOT)/src/caffetrt/wearmask_trt.cpp# Caffe推理网络，口罩检测部分
endif
#---------------

#----编译中间件----
#1.创建编译规则，将所有.cpp .cu文件编译成.o文件放置在OBJECTS_DIR中
OBJECT_FILE:=$(addprefix $(OBJECTS_DIR)/, $(addsuffix .o, $(basename $(notdir $(PROJECT_SRC)))))
#2.将编译规则存入rules中
RULEFILE:=$(OBJECTS_DIR)/rules
#----------------

#----编译中间件----
.PHONY:all
all: $(RULEFILE) $(OBJECT_FILE)

$(RULEFILE):
	@echo "Generating dependency file...";
	@-mkdir -p $(OBJECTS_DIR)
	@-rm -rf $(RULEFILE)
	@for f in $(PROJECT_SRC); do \
		OBJ=$(OBJECTS_DIR)/`basename $$f|sed -e 's/\.cpp/\.o/'`; \
		echo $$OBJ: $$f>> $(RULEFILE); \
		echo '	$(CC) $$(GCC_FLAGS) $$(DEFS) $$(INCPATHS) -c -o $$@ $$^'>> $(RULEFILE); \
	done
	@for g in $(PROJECT_SRC); do \
		OBJ=$(OBJECTS_DIR)/`basename $$g|sed -e 's/\.cu/\.o/'`; \
		echo $$OBJ: $$g>> $(RULEFILE); \
		echo '	$(NVCC) -std=c++11 -Xcompiler -fpic $$(INCPATHS) -c -o $$@ $$^'>> $(RULEFILE); \
	done
        
-include $(RULEFILE)
#-----------------

#----生成链接动态库----
.PHONY:libgpu
libgpu: $(RULEFILE) $(OBJECT_FILE)
	$(CC) -shared $(DEFS) $(LIBPATHS) $(OBJECT_FILE) -o $(PRJ_ROOT)/lib/$(LIB_NAME) -Xlinker --unresolved-symbols=ignore-in-shared-libs $(LIBS)
#-----------------

#----生成测试程序----
.PHONY:demo
demo: $(RULEFILE)
	$(CC) $(GCC_FLAGS) $(DEFS) $(LIBPATHS) $(OBJECT_FILE) -o $(PRJ_ROOT)/bin/$(DEMO_NAME) $(LIBS)
#-----------------

#----打包算法库----
INSTALL_LIB:=lib-$(LIB_VER)-$(SYS_VER)-$(CUDA_VER)-$(CUDNN_VER)-$(TRT_VER)-$(GPU_VER)
INSTALL_CONFIG:=config-$(MODEL_VER)

.PHONY: install
install:
	@rm -rf $(PRJ_ROOT)/package/TRT
	@mkdir -p $(PRJ_ROOT)/package/TRT/$(INSTALL_LIB)
	cp -rd $(PRJ_ROOT)/config $(PRJ_ROOT)/package/TRT/$(INSTALL_CONFIG)
	cd $(PRJ_ROOT)/package/TRT;ln -s $(INSTALL_CONFIG) config
	cp $(PRJ_ROOT)/lib/$(LIB_NAME) $(PRJ_ROOT)/package/TRT/$(INSTALL_LIB)/
	cd $(PRJ_ROOT)/package/TRT/$(INSTALL_LIB); ln -s $(LIB_NAME) libfaceVer.so
	cp -d $(TENSORRT_PATH)/lib/libnvinfer.so* $(PRJ_ROOT)/package/TRT/$(INSTALL_LIB)/
	cp -d $(TENSORRT_PATH)/lib/libnvparsers.so* $(PRJ_ROOT)/package/TRT/$(INSTALL_LIB)/
	cp -d $(TENSORRT_PATH)/lib/libnvinfer_plugin.so* $(PRJ_ROOT)/package/TRT/$(INSTALL_LIB)/
	cp -d $(TENSORRT_PATH)/lib/libnvonnxparser.so* $(PRJ_ROOT)/package/TRT/$(INSTALL_LIB)/
	cp -d $(TENSORRT_PATH)/lib/libmyelin.so* $(PRJ_ROOT)/package/TRT/$(INSTALL_LIB)/

	# cp -d $(CUDAS_PATH)/lib64/libcublas.so* $(PRJ_ROOT)/package/TRT/$(INSTALL_LIB)/
	# cp -d $(CUDAS_PATH)/lib64/libcudart.so* $(PRJ_ROOT)/package/TRT/$(INSTALL_LIB)/
	# cp -d $(CUDAS_PATH)/lib64/libnvrtc.so* $(PRJ_ROOT)/package/TRT/$(INSTALL_LIB)/
	# cp -d $(CUDAS_PATH)/lib64/libcudnn.so* $(PRJ_ROOT)/package/TRT/$(INSTALL_LIB)/

	cp -d $(FAISS_PATH)/libfaiss.* $(PRJ_ROOT)/package/TRT/$(INSTALL_LIB)/

	cd $(PRJ_ROOT)/package/TRT;ln -s $(INSTALL_LIB) lib


#-----------------

.PHONY: clean
clean: 
	rm -rf $(OBJECTS_DIR)
	rm -rf $(PRJ_ROOT)/package
	rm -rf $(PRJ_ROOT)/bin/*