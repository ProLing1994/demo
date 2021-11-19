#!/bin/bash

# 该脚本运行 kaldi 强制对齐工具进行强制对齐
# 作用：用于韵律节奏标注
# 注意：执行文本前，需要注意音频格式采样率为 16K，否则会报错

stage=2

config_file=/home/huanyuan/code/demo/Speech/TTS/config/tts/tts_config_chinese_sv2tts.py
# align_data_set=/mnt/huanyuan2/data/speech/tts/Chinese_dataset/dataset_align/BZNSYP_training
# align_data_set=/mnt/huanyuan2/data/speech/tts/Chinese_dataset/dataset_align/Aishell3_training
align_data_set=/mnt/huanyuan2/data/speech/tts/Chinese_dataset/dataset_align/Aishell3_testing
echo "script/dataset_align/dataset_align.sh"

# kaldi type data preparation
# 注意：代码需要根据唤醒词进行相应的更改，spk/device/text
if [ $stage -le 1 ];then
	python prepare_dataset_kaldi_type.py --config_file $config_file || exit 1
fi

# align words time index
if [ $stage -le 2 ];then
    echo "[Begin] align_nnet3_word"
    cd ./src
	echo $PWD
    ./align_nnet3_word.sh $align_data_set || exit 1
	cd ../
    echo "[Done] align_nnet3_word"
fi

echo "script/dataset_align/dataset_align.sh succeeded"