import argparse
import sys
from numba.core.types.misc import ClassDataType
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
# sys.path.insert(0, '/home/engineers/yh_rmai/code/demo/Speech')
from Basic.dataset import audio
from Basic.utils.hdf5_tools import *
from Basic.utils.folder_tools import *
from Basic.utils.train_tools import *

from SV.utils.infer_tools import *

from TTS.utils.tts.infer_tools import *
from TTS.utils.tts.visualizations_tools import *

from TTS.infer.tts.infer_tts import TtsSynthesizer
from TTS.dataset.tts.sv2tts_dataset_preload_audio_hdf5 import SynthesizerDataset

def infer(args):
    """
    模型推理
    """
    # load config
    cfg = load_cfg_file(args.config_file)

    # load tts synthesize
    tts_synthesize = TtsSynthesizer(args.config_file)
    tts_synthesize.load_speaker_info(args.speaker_id, args.speaker_wav)

    # load dataset
    dataset = SynthesizerDataset(cfg, hparams.TRAINING_NAME, bool_return_text=True)
    print("The number of files: {}".format(len(dataset)))

    for  text, wav, data_name in tqdm(dataset):
        # synthesize_spectrogram
        # mel: (C, T)
        mel, align_score = tts_synthesize.synthesize(text)
    
        ## Generating the waveform
        print("Synthesizing the waveform: ", text)
        print("Synthesizing Align Score: " , align_score)
    
        # save griffin lim inverted wav for debug (mel -> wav)
        wav_griffin = audio.compute_inv_mel_spectrogram(cfg, mel)

        # save 
        output_dir = os.path.join(cfg.general.save_dir, "wavs_test")
        create_folder(output_dir)
        wav_fpath = os.path.join(output_dir, data_name)
        audio.save_wav(wav_griffin, str(wav_fpath), sampling_rate=cfg.dataset.sampling_rate)

        # save statistics
        write_hdf5(
            os.path.join(output_dir, str(data_name).split('.')[0] + ".h5"),
            "wave",
            wav.astype(np.float32),
        )
        write_hdf5(
            os.path.join(output_dir, str(data_name).split('.')[0] + ".h5"),
            "feats",
            mel.T.astype(np.float32),   # mel: (T, C)
        )

def main():
    parser = argparse.ArgumentParser(description='Streamax TTS Training Engine')
    # chinese, prosody py
    parser.add_argument('-i', '--config_file', type=str, default="/mnt/huanyuan/model/tts/chinese_tts/sv2tts_chinese_new_tacotron2_singlespeaker_prosody_py_1_3_diff_feature_11292021/tts_config_chinese_sv2tts.py", nargs='?', help='config file')  # tacotron2 单说话人，韵律标签，与 vocoder 采用相同的特征，stop 预测位置优化，静音处优化
    parser.add_argument('-s', '--speaker_id', type=int, default=0, nargs='?', help='config file')
    parser.add_argument('-w', '--speaker_wav', type=str, default="/home/huanyuan/code/demo/Speech/TTS/infer/sample/Aishell3/SSB00050001.wav", nargs='?', help='config file')
    args = parser.parse_args()

    infer(args)

if __name__ == "__main__":
    main()