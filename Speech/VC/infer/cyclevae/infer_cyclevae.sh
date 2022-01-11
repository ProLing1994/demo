#!/bin/bash

#######################################
#           STAGE SETTING             #
#######################################
# 0: decoding conversion step [for extracting converted features for synthesis w/ PWG]
# 1: decoding reconstruction step [for training neural vocoder/Paralel WaveGAN (PWG)]
# 2: decoding cycle reconstruction step [for training neural vocoder/Paralel WaveGAN (PWG)]

stage=0

# init
config_file=/home/huanyuan/code/demo/Speech/VC/config/cyclevae/vc_config_cyclevae.py

echo "infer_cyclevae.sh"

######################################################
#                DECODING CONV. FEAT                 #
######################################################

echo "infer/cycelvae/infer_cyclevae.py"

if [ $stage -le 0 ];then
    python infer_cyclevae.py -i $config_file || exit 1
fi

echo "infer/cycelvae/infer_cyclevae.py succeeded"


#############################################################
#                  DECODING RECONST. FEAT                   #
#############################################################

echo "infer/cycelvae/infer_cyclevae_reconst.py"

if [ $stage -le 1 ];then
    python infer_cyclevae_reconst.py -i $config_file || exit 1
fi

echo "infer/cycelvae/infer_cyclevae_reconst.py succeeded"


#############################################################
#               DECODING CYCLIC RECONST. FEAT               #
#############################################################

echo "infer/cycelvae/infer_cyclevae_cycle_reconst.py"

if [ $stage -le 2 ];then
    python infer_cyclevae_cycle_reconst.py -i $config_file || exit 1
fi

echo "infer/cycelvae/infer_cyclevae_cycle_reconst.py succeeded"



echo "infer_cyclevae.sh succeeded"