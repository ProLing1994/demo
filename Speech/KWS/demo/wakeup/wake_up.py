from wakeup.utils import load_cfg_file
from wakeup.dataset_helper import load_label_index
from wakeup.pred_pyimpl import kws_load_model, model_predict
from wakeup.recognizer_pyimpl import RecognizeResult, RecognizeCommands


class WakeUp:
    """
    语音唤醒
    """
    def __init__(self, config_file="./wakeup/wake_up_config.py"):

        # load config
        self._cfg = load_cfg_file(config_file)

        # init dataset parameter 
        label_list = self._cfg.dataset.label.label_list
        self._positive_label_list = self._cfg.dataset.label.positive_label
        self._clip_duration_ms = self._cfg.dataset.clip_duration_ms
        self._sample_rate = self._cfg.dataset.sample_rate

        # inin model parameter
        model_path = self._cfg.general.model_path
        gpu_ids = int(self._cfg.general.gpu_ids)
        model_epoch = self._cfg.general.model_epoch
        
        # init test parameter 
        detection_threshold = self._cfg.test.detection_threshold
        timeshift_ms = self._cfg.test.timeshift_ms
        average_window_duration_ms = self._cfg.test.average_window_duration_ms

        # load label index 
        label_index = load_label_index(self._cfg.dataset.label.positive_label, self._cfg.dataset.label.negative_label)

        # init parameter
        self._audio_data = None
        self._audio_data_length = 0
        self._audio_data_offset = 0

        self._desired_samples = int(self._sample_rate * self._clip_duration_ms / 1000)
        self._timeshift_samples = int(self._sample_rate * timeshift_ms / 1000)

        # init recognizer
        self._recognize_element = RecognizeResult()
        self._recognize_commands = RecognizeCommands(
            labels=label_list,
            positove_lable_index = label_index[self._positive_label_list[0]],
            average_window_duration_ms=average_window_duration_ms,
            detection_threshold=detection_threshold,
            suppression_ms=3000,
            minimum_count=15)

        # init model
        model = kws_load_model(model_path, gpu_ids, model_epoch)
        self._net = model['prediction']['net']
        self._net.eval()

    def predict(self, input_data):

        # model infer
        output_score = model_predict(self._cfg, self._net, input_data)

        # process result
        current_time_ms = int(self._audio_data_offset * 1000 / self._sample_rate)
        self._recognize_commands.process_latest_result(output_score, current_time_ms, self._recognize_element)

        if self._recognize_element.is_new_command:
            return True
        return False


    @property
    def audio_data(self):
        return self._audio_data

    @audio_data.setter
    def audio_data(self, value):
        self._audio_data = value

    @property
    def audio_data_length(self):
        return self._audio_data_length

    @audio_data_length.setter
    def audio_data_length(self, value):
        self._audio_data_length = value

    @property
    def audio_data_offset(self):
        return self._audio_data_offset

    @audio_data_offset.setter
    def audio_data_offset(self, value):
        self._audio_data_offset = value

    @property
    def positive_label_list(self):
        return self._positive_label_list

    @property
    def sample_rate(self):
        return self._sample_rate

    @property
    def desired_samples(self):
        return self._desired_samples

    @property
    def timeshift_samples(self):
        return self._timeshift_samples