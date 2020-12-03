#!/bin/bash

# 该脚本运行 kaldi 强制对齐工具进行强制对齐
# 作用 1：后续用于语音切割
# 作用 2：后续用于基于帧对齐像素颗粒度的语音唤醒

stage=1

# init
data_set=/mnt/huanyuan/data/speech/kws/xiaorui_dataset/original_dataset/XiaoRuiDataset_12022020

echo "script/dataset_align/dataset_align.sh.sh"

# kaldi type data preparation
if [ $stage -le 1 ];then
	python prepare_dataset_kaldi_type.py --dataset_path $data_set --dest_path $data_set/kaldi_type --keyword_list "xiaorui" --keyword_chinese_name_list "小 锐 小 锐" || exit 1
fi

# align words time index
if [ $stage -le 2 ];then
    echo "[Begin] align_nnet3_word"
    cd ./src
	echo $PWD
	./align_nnet3_word.sh $data_set || exit 1
	cd ../
    echo "[Done] align_nnet3_word"
fi

# cut keyword data
if [ $stage -le 3 ];then
    python cut_keyword_wav.py --ctm_file $data_set/kaldi_type/tmp/nnet3_align/ctm --wav_file $data_set/kaldi_type/wav.scp --keyword_list "小,锐,小#,锐#" --save_dir $data_set/kaldi_cut_keyword
fi

echo "script/dataset_align/dataset_align.sh succeeded"