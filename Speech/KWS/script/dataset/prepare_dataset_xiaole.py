import argparse
import multiprocessing 
import os 
import shutil

from tqdm import tqdm


def get_hash_name(file_name):
    if "唤醒词" in file_name:
        hash_name = file_name.strip().split('.')[0].split('_')[0]
    elif 'XIAORUI' in file_name:
        hash_name = file_name.strip().split('_')[-1].split('.')[0][:6]
    elif "小乐小乐" in file_name:
        hash_name = file_name.strip().split('-')[0].split('_')[1]
    elif "XIAOYU" in file_name:
        hash_name = file_name.strip().split('_')[-1].split('.')[0][:6]
    else:
        hash_name = file_name.strip().split('.')[0].split('_')[0]
    return hash_name

def copy_other_file(args):
    file_path = args[0]
    output_other_dir = args[1]
    file_name = os.path.basename(file_path)
    if file_name.startswith('other_PVTC'):
        pst_path = os.path.join(output_other_dir, file_name)
    else:
        speaker = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(file_path))))
        device = os.path.basename(os.path.dirname(file_path))
        id = os.path.basename(file_path).split('.')[0]
        pst_name = "other_{}-other-{}-{}.wav".format(speaker, device, id)
        pst_path = os.path.join(output_other_dir, pst_name)
    tqdm.write("{} -> {}".format(file_path, pst_path))
    shutil.copy(file_path, pst_path)


def dataset_generator(input_dir, output_dir):
    # xiaole
    wav_file_path = os.path.join(input_dir, "PVTC", "wav.scp")
    wav_list = [line.split()[1] for line in open(wav_file_path)]
    print("Positive label: xiaole, number: ", len(wav_list))

    output_xiaole_dir = os.path.join(output_dir, "xiaole")
    # mkdir 
    if not os.path.exists(output_xiaole_dir):
        os.mkdir(output_xiaole_dir)

    for file_path in tqdm(wav_list):
        pst_path = os.path.join(output_xiaole_dir, os.path.basename(file_path))
        tqdm.write("{} -> {}".format(file_path, pst_path))
        shutil.copy(file_path, pst_path)

    # other
    wav_file_path = os.path.join(input_dir, "PVTC", "neg_wav.scp")
    wav_list = [line.split()[1] for line in open(wav_file_path)]
    print("Positive label: xiaole, number: ", len(wav_list))

    output_other_dir = os.path.join(output_dir, "other")
    # mkdir 
    if not os.path.exists(output_other_dir):
        os.mkdir(output_other_dir)

    in_params = []
    for file_path in tqdm(wav_list):
        in_args = [file_path, output_other_dir]
        in_params.append(in_args)

    p = multiprocessing.Pool(12)
    out = p.map(copy_other_file, in_params)
    p.close()
    p.join()

def main():
    parser = argparse.ArgumentParser(description="Prepare XiaoLe Dataset")
    parser.add_argument('--input_dir', type=str, default='/mnt/huanyuan/data/speech/kws/lenovo/lenovo_11242020/data/')
    parser.add_argument('--output_dir', type=str, default='/mnt/huanyuan/data/speech/kws/lenovo/LenovoDataset_11242020/')
    args = parser.parse_args()
    dataset_generator(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()