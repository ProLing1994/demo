import argparse
import collections
import pandas as pd
import pickle
import sys
import time 
import torch.nn.functional as F

from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo/Speech/KWS')
from utils.train_tools import *
from dataset.kws.dataset_helper import *
from impl.pred_pyimpl import kws_load_model, dataset_add_noise, model_predict
from script.analysis_result.plot_score_line import show_score_line
from script.analysis_result.cal_fpr_tpr import cal_fpr_tpr

class RecognizeResult(object):
    """Save recognition result temporarily.

    Attributes:
      founded_command: A string indicating the word just founded. Default value
        is '_silence_'
      score: An float representing the confidence of founded word. Default
        value is zero.
      is_new_command: A boolean indicating if the founded command is a new one
        against the last one. Default value is False.
    """

    def __init__(self):
        self._founded_command = SILENCE_LABEL
        self._score = 0
        self._is_new_command = False
        self._start_time = 0
        self._end_time = 0

    @property
    def founded_command(self):
        return self._founded_command

    @founded_command.setter
    def founded_command(self, value):
        self._founded_command = value

    @property
    def score(self):
        return self._score

    @score.setter
    def score(self, value):
        self._score = value

    @property
    def is_new_command(self):
        return self._is_new_command

    @is_new_command.setter
    def is_new_command(self, value):
        self._is_new_command = value

    @property
    def start_time(self):
        return self._start_time

    @start_time.setter
    def start_time(self, value):
        self._start_time = value

    @property
    def end_time(self):
        return self._end_time

    @end_time.setter
    def end_time(self, value):
        self._end_time = value


class RecognizeCommands(object):
    """Smooth the inference results by using average window.

    Maintain a slide window over the audio stream, which adds new result(a pair of
    the 1.confidences of all classes and 2.the start timestamp of input audio
    clip) directly the inference produces one and removes the most previous one
    and other abnormal values. Then it smooth the results in the window to get
    the most reliable command in this period.

    Attributes:
      _label: A list containing commands at corresponding lines.
      _average_window_duration: The length of average window.
      _detection_threshold: A confidence threshold for filtering out unreliable
        command.
      _suppression_ms: Milliseconds every two reliable founded commands should
        apart.
      _minimum_count: An integer count indicating the minimum results the average
        window should cover.
      _previous_results: A deque to store previous results.
      _label_count: The length of label list.
      _previous_top_label: Last founded command. Initial value is '_silence_'.
      _previous_top_time: The timestamp of _previous results. Default is -np.inf.
    """

    def __init__(self, labels, positove_lable_index,average_window_duration_ms, detection_threshold,
                 suppression_ms, minimum_count):
        """Init the RecognizeCommands with parameters used for smoothing."""
        # Configuration
        self._labels = labels
        self._positove_lable_index = positove_lable_index
        self._average_window_duration_ms = average_window_duration_ms
        self._detection_threshold = detection_threshold
        self._suppression_ms = suppression_ms
        self._minimum_count = minimum_count
        # Working Variable
        self._previous_results = collections.deque()
        self._label_count = len(labels)
        self._previous_top_time = 0

    def process_latest_result(self, latest_results, current_time_ms,
                              recognize_element):
        """Smoothing the results in average window when a new result is added in.

        Receive a new result from inference and put the founded command into
        a RecognizeResult instance after the smoothing procedure.

        Args:
          latest_results: A list containing the confidences of all labels.
          current_time_ms: The start timestamp of the input audio clip.
          recognize_element: An instance of RecognizeResult to store founded
            command, its scores and if it is a new command.

        Raises:
          ValueError: The length of this result from inference doesn't match
            label count.
          ValueError: The timestamp of this result is earlier than the most
            previous one in the average window
        """
        if latest_results[0].shape[0] != self._label_count:
            raise ValueError("The results for recognition should contain {} "
                             "elements, but there are {} produced".format(
                                 self._label_count, latest_results[0].shape[0]))
        if (self._previous_results.__len__() != 0 and
                current_time_ms < self._previous_results[0][0]):
            raise ValueError("Results must be fed in increasing time order, "
                             "but receive a timestamp of {}, which was earlier "
                             "than the previous one of {}".format(
                                 current_time_ms, self._previous_results[0][0]))

        # Add the latest result to the head of the deque.
        self._previous_results.append([current_time_ms, latest_results[0]])

        # Prune any earlier results that are too old for the averaging window.
        time_limit = current_time_ms - self._average_window_duration_ms
        while time_limit > self._previous_results[0][0]:
            self._previous_results.popleft()

        # If there are too few results, the result will be unreliable and bail.
        how_many_results = self._previous_results.__len__()
        earliest_time = self._previous_results[0][0]
        sample_duration = current_time_ms - earliest_time
        if (how_many_results < self._minimum_count or
                sample_duration < self._average_window_duration_ms / 4):
            recognize_element.score = 0.0
            recognize_element.is_new_command = False
            recognize_element.founded_command = SILENCE_LABEL
            return

        # Calculate the average score across all the results in the window.
        average_scores = np.zeros(self._label_count)
        for item in self._previous_results:
            score = item[1]
            for i in range(score.size):
                average_scores[i] += score[i] / how_many_results
        recognize_element.score = average_scores[self._positove_lable_index]

        time_since_last_top = current_time_ms - self._previous_top_time
        if self._previous_top_time == 0:
            time_since_last_top = np.inf

        if (recognize_element.score > self._detection_threshold and
                time_since_last_top > self._suppression_ms):
            self._previous_top_time = current_time_ms
            recognize_element.is_new_command = True
            recognize_element.start_time = self._previous_results[0][0]
            recognize_element.end_time = self._previous_results[-1][0]
            recognize_element.founded_command = self._labels[self._positove_lable_index]
        else:
            recognize_element.is_new_command = False        
            recognize_element.founded_command = SILENCE_LABEL


def test(input_wav, config_file, model_epoch, timeshift_ms, average_window_duration_ms, detection_threshold):
    # load configuration file
    cfg = load_cfg_file(config_file)

    # init
    sample_rate = cfg.dataset.sample_rate
    clip_duration_ms = cfg.dataset.clip_duration_ms
    desired_samples = int(sample_rate * clip_duration_ms / 1000)
    timeshift_samples = int(sample_rate * timeshift_ms / 1000)
    label_list = cfg.dataset.label.label_list
    positive_label = cfg.dataset.label.positive_label

    # load label index 
    label_index = load_label_index(cfg.dataset.label.positive_label)

    recognize_element = RecognizeResult()
    recognize_commands = RecognizeCommands(
        labels=label_list,
        positove_lable_index = label_index[positive_label[0]],
        average_window_duration_ms=average_window_duration_ms,
        detection_threshold=detection_threshold,
        suppression_ms=3000,
        minimum_count=15)
    
    # mkdir 
    # output_dir = os.path.join(os.path.dirname(input_wav), os.path.basename(input_wav).split('.')[0])
    output_dir = os.path.join(cfg.general.save_dir, 'test_straming_wav', os.path.basename(input_wav).split('.')[0])
    if not os.path.exists(output_dir):    
        os.makedirs(output_dir)
    
    # load model
    model = kws_load_model(cfg.general.save_dir, int(
        cfg.general.gpu_ids), model_epoch)
    net = model['prediction']['net']
    net.eval()

    # load data
    audio_data = librosa.core.load(input_wav, sr=sample_rate)[0]
    assert len(audio_data) > desired_samples, "[ERROR:] Wav is too short! Need more than {} samples but only {} were found".format(
        desired_samples, len(audio_data))

    audio_data_offset = 0
    original_scores = [] 
    mean_scores = [] 
    all_found_words = []

    # record tiem
    start = time.perf_counter()

    while(audio_data_offset < len(audio_data)):
        print('Done : [{}/{}]'.format(audio_data_offset, len(audio_data)),end='\r')

        # input data
        input_start = audio_data_offset
        input_end = audio_data_offset + desired_samples
        input_data = audio_data[input_start: input_end]
        if len(input_data) != desired_samples:
            break
        
        audio_data_offset += timeshift_samples

        # model infer
        output_score = model_predict(cfg, net, input_data)

        # process result
        current_time_ms = int(input_start * 1000 / sample_rate)
        recognize_commands.process_latest_result(output_score, current_time_ms, recognize_element)

        if recognize_element.is_new_command:
            all_found_words_dict = {}
            all_found_words_dict['label'] = positive_label[0]
            all_found_words_dict['start_time'] = recognize_element.start_time
            all_found_words_dict['end_time'] = recognize_element.end_time + clip_duration_ms
            all_found_words.append(all_found_words_dict)
            print('Find words: label:{}, start time:{}, end time:{}'.format(all_found_words_dict['label'], all_found_words_dict['start_time'], all_found_words_dict['end_time']))

            if bool_write_audio:
                output_path = os.path.join(output_dir, 'label_{}_starttime_{}.wav'.format(all_found_words_dict['label'], all_found_words_dict['start_time']))
                start_time = int(sample_rate * all_found_words_dict['start_time'] / 1000)
                end_time = int(sample_rate * all_found_words_dict['end_time'] / 1000)
                output_wav = audio_data[start_time: end_time]
                librosa.output.write_wav(output_path, output_wav, sr=sample_rate)

        original_scores.append({'start_time':current_time_ms, 'score':output_score[0][label_index[positive_label[0]]]})
        mean_scores.append({'start_time':current_time_ms, 'score':recognize_element.score})

    # record time
    end = time.perf_counter()
    print('Running time: %s Seconds'%(end - start))

    found_words_pd = pd.DataFrame(all_found_words)
    found_words_pd.to_csv(os.path.join(output_dir, 'found_words.csv'), index=False)
    original_scores_pd = pd.DataFrame(original_scores)
    original_scores_pd.to_csv(os.path.join(output_dir, 'original_scores.csv'), index=False)
    mean_scores_pd = pd.DataFrame(mean_scores)
    mean_scores_pd.to_csv(os.path.join(output_dir, 'mean_scores.csv'), index=False)
    
    # show result
    show_score_line(input_wav.split('.')[0] + '.csv', os.path.join(output_dir, 'original_scores.csv'), positive_label[0])
    show_score_line(input_wav.split('.')[0] + '.csv', os.path.join(output_dir, 'mean_scores.csv'), positive_label[0])

    cal_fpr_tpr(input_wav.split('.')[0] + '.csv', os.path.join(output_dir, 'found_words.csv'),  positive_label[0], bool_write_audio)

def main():
    """
    使用模型对音频文件进行测试，模拟真实音频输入情况，配置为 --input 中的 config 文件，该脚本会通过滑窗的方式测试每一小段音频数据，计算连续 2000ms(41帧) 音频的平均值结果，
    如果超过预设门限，则认为检测到关键词，否则认定未检测到关键词，最后分别计算假阳性和召回率
    """
    # default_input_wav = "/home/huanyuan/model/test_straming_wav/xiaoyu_03022018_training_60_001.wav"
    # default_input_wav = "/home/huanyuan/model/test_straming_wav/xiaoyu_03022018_validation_60_001.wav"
    # default_input_wav = "/home/huanyuan/model/test_straming_wav/xiaoyu_03022018_testing_60_001.wav"
    # default_input_wav = "/home/huanyuan/model/test_straming_wav/weiboyulu_test_3600_001.wav"
    # default_input_wav = "/home/huanyuan/model/test_straming_wav/xiaoyu_03022018_testing_3600_001.wav"
    default_input_wav = "/home/huanyuan/model/test_straming_wav/xiaoyu_10292020_testing_3600_001.wav"

    # defaule_config_file = "/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_xiaoyu.py"
    defaule_config_file = "/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_xiaoyu_2.py"
    default_model_epoch = -1
    # default_timeshift_ms = 30
    # default_average_window_duration_ms = 800
    default_timeshift_ms = 50
    default_average_window_duration_ms = 2000
    # default_detection_threshold = 0.8
    default_detection_threshold = 0.95

    parser = argparse.ArgumentParser(description='Streamax KWS Testing Engine')
    parser.add_argument('--input_wav', type=str,
                        default=default_input_wav)
    parser.add_argument('--config_file', type=str,
                        default=defaule_config_file)
    parser.add_argument('--model_epoch', type=str, default=default_model_epoch)
    parser.add_argument('--timeshift_ms', type=int,
                        default=default_timeshift_ms)
    parser.add_argument('--average_window_duration_ms',
                        type=int, default=default_average_window_duration_ms)
    parser.add_argument('--detection_threshold',
                        type=int, default=default_detection_threshold)
    args = parser.parse_args()

    test(args.input_wav, args.config_file, args.model_epoch,
         args.timeshift_ms, args.average_window_duration_ms, args.detection_threshold)


if __name__ == "__main__":
    bool_write_audio = True
    main()
