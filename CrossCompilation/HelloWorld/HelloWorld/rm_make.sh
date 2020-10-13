#!/bin/bash

_DEF_FILE_PATH_=$(pwd)/defines

if [ -f ${_DEF_FILE_PATH_} ]; then
  rm ${_DEF_FILE_PATH_}
fi

echo "export RELEASE_ROOT_DIR:=$(pwd)/release" >> $_DEF_FILE_PATH_;

PRODUCT_LIST=(
"ADkit_3516DV300"
"ADkit_3519AV100"
)
echo "product:"
for i in "${!PRODUCT_LIST[@]}"; do
	echo "$i: ${PRODUCT_LIST[$i]}"
done
echo -ne "please input product id:"
read _product_;

if [ "${_product_}" -lt "${#PRODUCT_LIST[@]}" ];then
	PRODUCT_TYPE=${PRODUCT_LIST[${_product_}]%(*}
	echo "export PRODUCT_TYPE:=${PRODUCT_TYPE}" >> $_DEF_FILE_PATH_;

	for c in ${PRODUCT_TYPE}
	do
		case $c in	
		  ADkit_3516DV300)
		    echo "export CROSS_COMPILER:=arm-himix200-linux-" >> $_DEF_FILE_PATH_;
		    echo "export COMPILER_NAME:=arm-himix200-linux" >> $_DEF_FILE_PATH_;
		    ;;
		  ADkit_3519AV100)
		    echo "export CROSS_COMPILER:=arm-himix200-linux-" >> $_DEF_FILE_PATH_;
		    echo "export COMPILER_NAME:=arm-himix200-linux" >> $_DEF_FILE_PATH_;
		    ;;
		  *)
		    echo "invalid product:${PRODUCT_TYPE}";	
		    ;;
		esac
	done
else
	echo "invalid product!"
	exit 1
fi

echo "version type:";
echo " 1: release";
echo " 2: debug";
echo -ne "please input version type id:"
read _version_;

if [ "$_version_" = "2" ];then
	echo "export DEBUG_VERSION:=y" >> $_DEF_FILE_PATH_;
else
	echo "export DEBUG_VERSION:=n" >> $_DEF_FILE_PATH_;
fi