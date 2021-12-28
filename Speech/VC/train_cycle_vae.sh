#!/bin/bash

#######################################
#           STAGE SETTING             #
#######################################
# 0: data list preparation step
# 1: feature extraction step
# 2: statistics calculation step
# 3: training step

stage=3

echo "train_cycle_vae.sh"

###########################################################
#                 DATA PREPARATION STEP                   #
###########################################################
if [ $stage -le 0 ];then
    cd ./script/dataset/
    ./data_train_test_split.sh || exit 1
    cd ../..
fi

###########################################################
#               FEATURE EXTRACTION STEP                   #
###########################################################

if [ $stage -le 1 ];then
    cd ./script/dataset/
    ./data_preload_audio_hdf5.sh || exit 1
    cd ../..
fi

# ###########################################################
# #            CALCULATE SPEAKER STATISTICS STEP            #
# ###########################################################

if [ $stage -le 2 ];then
    cd ./script/dataset/
    ./data_normalize_state.sh || exit 1
    cd ../..
fi

###############################################
#               TRAINING STEP                 #
###############################################

if [ $stage -le 3 ];then
    python train_cycle_vae.py -i $config_file || exit 1
fi

echo "train_cycle_vae.sh succeeded"