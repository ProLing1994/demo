import os
import pandas as pd
import shutil

def get_sub_filepaths_suffix(folder, suffix='.wav'):
    paths = []
    for root, dirs, files in os.walk(folder):
        for name in files:
            if not name.endswith(suffix):
                continue
            path = os.path.join(root, name)
            paths.append(path)
    return paths

def copy_folder():
    input_dir = "D:\\data\\test\\1113-1114"
    output_dir = "D:\\data\\test\\1113-1114-mp4"
    file_type = ".mp4"

    file_list = get_sub_filepaths_suffix(input_dir, file_type)
    file_list.sort()

    for idx in range(len(file_list)):
        input_path = file_list[idx] 
        output_path = os.path.join(output_dir, os.path.basename(input_path))
        print(input_path, '->', output_path)
        shutil.copy(input_path, output_path)
    

if __name__ == '__main__':
    copy_folder()
