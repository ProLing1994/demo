PROJECT_DIR=$(shell pwd)
MAKE_SUB_DIR=${PROJECT_DIR}

define MAKE_SUB_PROJECT
	@for subdir in $(MAKE_SUB_DIR); do \
		(echo "$$subdir" && cd $$subdir && make -f rm_makefile $1) \
	done
endef

ifndef NO_DEFINES
-include ./defines
endif

.PHONY:all
all:
	$(call MAKE_SUB_PROJECT,CREATE_RELY_FILE)
	$(call MAKE_SUB_PROJECT,all)

.PHONY:install
install:
	$(call MAKE_SUB_PROJECT,CREATE_RELY_FILE)
	$(call MAKE_SUB_PROJECT,all)

.PHONY:clean
clean:
	$(call MAKE_SUB_PROJECT,clean)
