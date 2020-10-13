PROJECT_DIR=$(shell pwd)
MAKE_SUB_DIR+=${PROJECT_DIR}/app


define MAKE_SUB_PROJECT
	@for subdir in $(MAKE_SUB_DIR); do \
		cd $$subdir; \
		mkdir -p ${RELEASE_ROOT_DIR}/$$(basename $$subdir); \
		make $1 NO_DEFINES=y PREFIX=${RELEASE_ROOT_DIR}/$$(basename $$subdir); \
	done
endef

ifndef NO_DEFINES
-include ./defines
endif

.PHONY:all
all:
	$(call MAKE_SUB_PROJECT,all)

.PHONY:install
install:
	$(call MAKE_SUB_PROJECT,all)

.PHONY:clean
clean:
	$(call MAKE_SUB_PROJECT,clean)
