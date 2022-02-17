import argparse
import numpy as np
import os
from tqdm import tqdm


def gen_txt(args):
    jpg_list = np.array(os.listdir(args.jpg_dir))
    jpg_list = jpg_list[[jpg.endswith('.jpg') for jpg in jpg_list]]
    jpg_list.sort()

    with open(args.output_path, 'w') as f:
        for idx in tqdm(range(len(jpg_list))):
            f.writelines("{}\n".format(jpg_list[idx].split('.')[0]))

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.jpg_dir = "/yuanhuan/data/image/Distance_detection/"
    args.output_path = os.path.join(args.jpg_dir, "images.txt")

    gen_txt(args)
