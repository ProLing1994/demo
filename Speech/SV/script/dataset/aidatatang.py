import os
import sys
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
from Basic.utils.folder_tools import *

if __name__ == "__main__":
    path = "/mnt/huanyuan/data/speech/asr/Chinese/aidatatang_200zh/aidatatang_200zh/corpus"
    file_path = get_sub_filepaths_suffix(path, 'tar.gz')

    for idx in tqdm(range(len(file_path))):
        file_idx = file_path[idx]
        os.system("tar -zxvf {} -C {}".format(file_idx, os.path.dirname(file_idx)))
        os.system("rm -f {}".format(file_idx))
