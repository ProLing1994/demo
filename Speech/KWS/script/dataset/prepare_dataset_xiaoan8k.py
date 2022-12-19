import argparse
import glob
import os
import shutil
import re


def get_hash_name(file_name):
    if "唤醒词" in file_name:
        hash_name = file_name.strip().split('.')[0].split('_')[0]
    elif 'XIAORUI' in file_name:
        hash_name = file_name.strip().split('_')[-1].split('.')[0][:6]
    elif "小乐小乐" in file_name:
        hash_name = file_name.strip().split('-')[0].split('_')[1]
    elif "XIAOYU" in file_name:
        hash_name = file_name.strip().split('_')[-1].split('.')[0][:6]
    elif "RM_KWS_XIAOAN_xiaoan_" in file_name:
        hash_name = file_name.strip().split('_')[-1].split('.')[0][:6]
    elif "RM_KWS_NIHAOXIAOAN_nihaoxiaoan_" in file_name:
        hash_name = file_name.strip().split('_')[-1].split('.')[0][:6]
    else:
        hash_name = file_name.strip().split('.')[0]
    return hash_name