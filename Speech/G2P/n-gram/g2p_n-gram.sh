#!/bin/bash

# data=mini-train
data=train

# LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib:/usr/local/lib/fst

# 训练
# dataset init 
python chinese_dataset.py dataset/${data}.dict dataset/${data}.formatted.corpus

# cd 
cd ./dataset

# Train an n-gram model (5s-10s):
estimate-ngram -o 8 -t ${data}.formatted.corpus \
  -wl ${data}.o8.arpa

# Convert to OpenFst format (10s-20s):
phonetisaurus-arpa2wfst --lm=${data}.o8.arpa --ofile=${data}.o8.fst

# 解码
phonetisaurus-apply --model ${data}.o8.fst --word_list test.dict > test_${data}.out

# 准确率
# cd 
cd ../

python acc.py --src_path dataset/test_${data}.out --gt_path dataset/gt.dict