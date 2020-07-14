ROOT_DIR:=$(shell pwd)
TARGET:= 

CC		:= g++ 
AR		:= ar
CFLAGS	:= -g -W -Wall -std=c++11 -DNNIE -fpic -fno-strict-aliasing -traditional -Wno-reorder -Wno-char-subscripts
INCLUDE	:= 
LIBS	:=
STATIC_LIB:=
LIBS	+= $(STATIC_LIB)

SRC_DIR	+=$(ROOT_DIR)/json
SRC_FILE_CPP+=$(foreach d, $(SRC_DIR), $(wildcard $(addprefix $(d)\/*, .cpp))) 
INCLUDE_FILE+=$(foreach d, $(SRC_DIR), $(wildcard $(addprefix $(d)\/*, .h))) 
SOURCE  +=$(SRC_FILE_CPP)

OBJS    := $(patsubst %.c, %.o, $(patsubst %.cpp, %.o, $(SOURCE)))

ifndef PREFIX
	RELEASE_DIR:=${ROOT_DIR}/release
else
	RELEASE_DIR:=${PREFIX}
endif

ifeq ($(suffix $(TARGET)), .a)
	IS_LIB:=TURE
endif

ifeq ($(suffix $(TARGET)), .so)
	IS_LIB:=TURE
endif

.PHONY:all
all:
	@echo ${RELEASE_DIR}
	@mkdir -p ${RELEASE_DIR}/json
	@cp -f $(INCLUDE_FILE) ${RELEASE_DIR}/json

%.o : %.cpp
	$(CC) $(CFLAGS) $(INCLUDE) -c -o $@ $<

.PHONY:clean
clean:
	rm -rf ${RELEASE_DIR}/json