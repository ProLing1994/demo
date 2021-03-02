# 注：请执行操作顺序：
# 1. 执行本文件，根据 csv 文件将音频进行划分，这里只划分单类音频，多个类别音频不进行处理

import argparse
import glob
import os 
import pandas as pd
import yaml

from tqdm import tqdm

leaf_nodes = ['Accelerating_and_revving_and_vroom', 'Accordion', 'Acoustic_guitar', 'Applause', 'Bark', 'Bass_drum', 'Bass_guitar',
                'Bathtub_(filling_or_washing)', 'Bicycle_bell', 'Boat_and_Water_vehicle', 'Boiling', 'Boom', 'Bowed_string_instrument',
                'Burping_and_eructation', 'Bus', 'Buzz', 'Camera', 'Car_passing_by', 'Chatter', 'Cheering', 'Chewing_and_mastication',
                'Chicken_and_rooster', 'Child_speech_and_kid_speaking', 'Chink_and_clink', 'Chirp_and_tweet', 'Chuckle_and_chortle',
                'Church_bell', 'Clapping', 'Coin_(dropping)', 'Computer_keyboard', 'Conversation', 'Cough', 'Cowbell', 'Crack', 'Crack',
                'Crackle', 'Crash_cymbal', 'Cricket', 'Crow', 'Crowd', 'Crumpling_and_crinkling', 'Crushing', 'Crying_and_sobbing', 
                'Cupboard_open_or_close', 'Cutlery_and_silverware', 'Dishes_and_pots_and_pans', 'Doorbell', 'Drawer_open_or_close', 'Drill',
                'Drip', 'Drum_kit', 'Electric_guitar', 'Engine_starting', 'Fart', 'Female_singing', 'Female_speech_and_woman_speaking', 'Fill_(with_liquid)',
                'Finger_snapping', 'Fireworks', 'Fixed-wing_aircraft_and_airplane', 'Frog', 'Frying_(food)', 'Gasp', 'Giggle', 'Glockenspiel',
                'Gong', 'Growling', 'Gull_and_seagull', 'Gunshot_and_gunfire', 'Gurgling', 'Hammer', 'Harmonica', 'Harp', 'Hi-hat', 'Hiss',
                'Idling', 'Keys_jangling', 'Knock', 'Livestock_and_farm_animals_and_working_animals', 'Male_singing', 'Male_speech_and_man_speaking',
                'Marimba_and_xylophone', 'Mechanical_fan', 'Meow', 'Microwave_oven', 'Motorcycle', 'Organ', 'Packing_tape_and_duct_tape', 'Piano',
                'Printer', 'Purr', 'Race_car_and_auto_racing', 'Raindrop', 'Ratchet_and_pawl', 'Rattle', 'Rattle_(instrument)', 'Ringtone',
                'Run', 'Sawing', 'Scissors', 'Scratching_(performance technique)', 'Screaming', 'Screech', 'Shatter', 'Sigh', 'Sink_(filling_or_washing)', 
                'Siren', 'Skateboard', 'Slam', 'Sliding door', 'Snare drum', 'Sneeze', 'Speech_synthesizer', 'Splash_and_splatter', 'Squeak',
                'Stream', 'Strum', 'Subway_and_metro_and_underground', 'Tabla', 'Tambourine', 'Tap', 'Tearing', 'Thump_and_thud', 'Thunder',
                'Tick', 'Tick-tock', 'Toilet_flush', 'Traffic_noise_and_roadway_noise', 'Train', 'Trickle_and_dribble', 'Truck', 'Trumpet',
                'Typewriter', 'Vehicle_horn_and_car_horn_and_honking', 'Walk_and_footsteps', 'Water_tap_and_faucet', 'Waves_and_surf', 'Whispering',
                'Whoosh_and_swoosh_and_swish,Gunshot_and_gunfir', 'Wind', 'Wind_chime', 'Wind_instrument_and_woodwind_instrument', 'Writing',
                'Yell', 'Zipper_(clothing)']

intermediate_nodes = ['Aircraft', 'Alarm', 'Animal', 'Bell', 'Bicycle', 'Bird', 'Bird_vocalization_and_bird_call_and_bird_song',
                        'Brass_instrument', 'Breathing', 'Car', 'Cat', 'Chime', 'Clock', 'Cymbal', 'Dog', 'Domestic_animals_and_pets',
                        'Domestic_sounds_and_home_sounds', 'Door', 'Drum', 'Engine', 'Explosion', 'Fire', 'Fowl', 'Glass', 'Guitar',
                        'Hands', 'Human_group_actions', 'Human_voice', 'Insect', 'Keyboard_(musical)', 'Laughter', 'Liquid', 'Mallet_percussion',
                        'Mechanisms', 'Motor_vehicle_(road)', 'Music', 'Musical_instrumen', 'Ocean', 'Percussion', 'Plucked_string_instrument',
                        'Pour', 'Power_tool', 'Rail_transport', 'Rain', 'Respiratory_sounds', 'Shout', 'Singing', 'Speech', 'Telephone', 'Thunderstorm',
                        'Tools', 'Typing', 'Vehicle', 'Vehicle', 'Water', 'Wild_animals', 'Wood']

def main():
    pd_csv = pd.read_csv(args.csv_path)

    # mkdir
    leaf_folder_name = 'leaf'
    intermediate_folder_name = 'intermediate'
    if not os.path.exists(os.path.join(args.output_folder, leaf_folder_name)):
        os.makedirs(os.path.join(args.output_folder, leaf_folder_name))
    if not os.path.exists(os.path.join(args.output_folder, intermediate_folder_name)):
        os.makedirs(os.path.join(args.output_folder, intermediate_folder_name))

    label_set = set()
    for _, row in tqdm(pd_csv.iterrows()):
        file_name = row["fname"]
        file_path = os.path.join(args.input_folder, str(file_name) + args.audio_suffix)
        assert os.path.exists(file_path)

        file_label_list = row['labels'].split(',')
        # print(file_label_list)

        intermediate_num = 0 
        intermediate_node = ""
        leaf_num = 0
        leaf_node = ""
        for idx in range(len(file_label_list)):
            if file_label_list[idx] in intermediate_nodes:
                intermediate_node = file_label_list[idx] 
                intermediate_num += 1
            if file_label_list[idx] in leaf_nodes:
                leaf_node = file_label_list[idx] 
                leaf_num += 1

        if intermediate_num == 1 or leaf_num == 1:
            if leaf_num == 1:
                print("leaf", leaf_node, file_label_list)
                output_dir = os.path.join(args.output_folder, leaf_folder_name, leaf_node)
                # mkdir
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                output_path = os.path.join(output_dir, os.path.basename(file_path))
                os.system("cp {} {}".format(file_path, output_path))

            elif intermediate_num == 1:
                print("intermediate", intermediate_node, file_label_list)
                output_dir = os.path.join(args.output_folder, intermediate_folder_name, intermediate_node)
                # mkdir
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                output_path = os.path.join(output_dir, os.path.basename(file_path))
                os.system("cp {} {}".format(file_path, output_path))

        for idx in range(len(file_label_list)):
            if file_label_list[idx] in args.special_label:
                output_dir = os.path.join(args.output_folder, "Total_{}".format(args.special_label))
                # mkdir
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                output_path = os.path.join(output_dir, os.path.basename(file_path))
                os.system("cp {} {}".format(file_path, output_path))
                label_set = label_set | set(file_label_list)
                break
    print(label_set)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FSD50K Engine')
    # parser.add_argument('--input_folder', type=str, default="/mnt/huanyuan/data/speech/sed/FSD50K/original_dataset/FSD50K.dev_audio/")
    parser.add_argument('--input_folder', type=str, default="/mnt/huanyuan/data/speech/sed/FSD50K/original_dataset/FSD50K.eval_audio/")
    parser.add_argument('--output_folder', type=str, default="/mnt/huanyuan/data/speech/sed/FSD50K/processed_dataset/")
    # parser.add_argument('--csv_path', type=str, default="/mnt/huanyuan/data/speech/sed/FSD50K/original_dataset/FSD50K.ground_truth/dev.csv")
    parser.add_argument('--csv_path', type=str, default="/mnt/huanyuan/data/speech/sed/FSD50K/original_dataset/FSD50K.ground_truth/eval.csv")
    parser.add_argument('--special_label', type=str, default="Gunshot_and_gunfire")
    parser.add_argument('--audio_suffix', type=str, default=".wav")
    args = parser.parse_args()

    main()