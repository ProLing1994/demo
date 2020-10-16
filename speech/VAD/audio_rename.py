import argparse
import os
import pandas as pd 
import shutil
import re

parser = argparse.ArgumentParser(description="Audio Rename")
parser.add_argument('--dir', type=str, default='E:\\project\\data\\weiboyulu\\1012\\0000000000000000-201012-103547-114012-000001089960')
args = parser.parse_args()

if __name__ == "__main__":
    csv_path = os.path.join(args.dir, os.path.basename(args.dir) + '.csv')
    csv_rename_path = os.path.join(args.dir, os.path.basename(args.dir) + '-rename.csv')
    assert os.path.exists(csv_path), "[ERROR:] csv file do not exist, Please check again!"
    assert not os.path.exists(csv_rename_path), "[ERROR:] You already renamed audio, Please do not rename again!"

    csv_pd = pd.read_csv(csv_path)

    # Check 
    # Check Input
    for idx, row in csv_pd.iterrows():
        if str(row.state) == 'N':
            continue
        elif str(row.state) == 'D':
            continue 
        elif len(str(row.state).strip().split('_')) == 4:
            if re.match(r'^S\d{3}M\d{1}P\d{5}', str(row.state).strip().split('_')[-1]):
                continue
            else:
                raise Exception("[ERROR:] Invalid input: audio_region: {}, state: {}".format(row.audio_region, row.state))
        else:
            raise Exception("[ERROR:] Invalid input: audio_region: {}, state: {}".format(row.audio_region, row.state))
    
    # Check Duplicate file 
    remainder_files_list = []
    for idx, row in csv_pd.iterrows():
        if str(row.state) == 'N':
            remainder_files_list.append(row.audio_region)
        elif str(row.state) == 'D':
            continue
        elif len(str(row.state).strip().split('_')) == 4:
            remainder_files_list.append(row.state)
        else:
            raise Exception("[ERROR:] Invalid input: audio_region: {}, state: {}".format(row.audio_region, row.state))

    remainder_files_set = list(set(remainder_files_list))
    if len(remainder_files_list) != len(remainder_files_set):
        for idx in range(len(remainder_files_set)):
            remainder_files_list.remove(remainder_files_set[idx])
        for idx in range(len(remainder_files_list)):
            print("[ERROR:] Invalid input: audio {} is Repetied, Please check again!".format(remainder_files_list[idx]))
        raise Exception("[ERROR:] Invalid input: audio is Repetied, Please check again!")
        

    print("Delete: ")
    # Delete
    for idx, row in csv_pd.iterrows():
        if str(row.state) == 'D':
            if os.path.exists(os.path.join(args.dir, row.audio_region + '.wav')):
                os.remove(os.path.join(args.dir, row.audio_region + '.wav'))
                print("Audio region remove: {}".format(row.audio_region + '.wav'))

    print("Delete Done")
    print()

    print("Rename")
    # rename 
    tmp_rename_files = []
    for idx, row in csv_pd.iterrows():
        if len(str(row.state).strip().split('_')) == 4:
            if re.match(r'^S\d{3}M\d{1}P\d{5}', str(row.state).strip().split('_')[-1]):
                if os.path.exists(os.path.join(args.dir, row.audio_region + '.wav')):
                    tmp_rename_files.append({'audio_region':row.audio_region, 'state':row.state, 'tmp':'tmp_' + row.state + '.wav'})
                    os.rename(os.path.join(args.dir, row.audio_region + '.wav'), os.path.join(args.dir, 'tmp_' + row.state + '.wav'))
                else:
                    raise Exception("[ERROR:] Audio: {} do not exist, please check!".format(row.audio_region + '.wav'))
    
    tmp_rename_files_pd = pd.DataFrame(tmp_rename_files)
    for idx, row in tmp_rename_files_pd.iterrows():
        os.rename(os.path.join(args.dir, 'tmp_' + row.state + '.wav'), os.path.join(args.dir, row.state + '.wav'))
        print("Audio region rename: {} -> {}".format(row.audio_region + '.wav', row.state + '.wav'))

    print("Rename Done")
    print()

    print("Rename csv file")
    # update 
    audio_region_list = []
    for idx, row in csv_pd.iterrows():
        audio_region_dict = {}
        if str(row.state) == 'N':
            audio_region_dict['audio_region'] = row.audio_region
            audio_region_dict['state'] = 'N'
        elif len(str(row.state).strip().split('_')) == 4:
            audio_region_dict['audio_region'] = row.state
            audio_region_dict['state'] = 'N'
        else:
            continue
        audio_region_list.append(audio_region_dict)
    audio_region_pd = pd.DataFrame(audio_region_list)
    audio_region_pd.to_csv(csv_rename_path, index=False)