#!/bin/bash

# 该脚本进行模型测试

stage=1

# init
# gorila
config_file=/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_gorila8k.py
model_epoch=-1

echo "test.sh"

# test model
if [ $stage -le 1 ];then
    python ./infer/infer.py --input $config_file || exit 1
fi

echo "test.sh succeeded"