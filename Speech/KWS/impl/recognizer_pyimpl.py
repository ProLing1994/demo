from dataset.kws.dataset_helper import *
import collections
import sys
import time

sys.path.insert(0, '/home/huanyuan/code/demo/Speech/KWS')


class DoubleEdgeDetecting(object):
    def __init__(self, detection_threshold_low, detection_threshold_high):
        """Init the DoubleEdgeDetecting with parameters."""
        # Configuration
        self._detection_threshold_low = detection_threshold_low
        self._detection_threshold_high = detection_threshold_high
        self._detection_threshold_highest = 0.9
        self._boundary_threshold = 0.02
        # Working Variable
        self._scores = collections.deque()

    def process_latest_result(self, score, deque_length):
        # init 
        detection_bool = False

        # Add the latest score to the head of the deque.
        self._scores.append(score)
        
        if len(self._scores) < deque_length:
            return detection_bool

        # Prune any earlier scores that are too old for the window.
        while deque_length < len(self._scores):
            self._scores.popleft()

        # 当得分超过最大阈值时，认为检测到结果
        if np.array(self._scores).max() > self._detection_threshold_highest:
            detection_bool = True
            return detection_bool

        # 双门限法，检测两个边缘，同时两个边缘分别大于两个预设门限值
        # 求一阶导数
        first_order_score_float = abs(np.array(self._scores)[1:] - np.array(self._scores)[:-1])
        first_order_score_binary = np.array([1 if x >= self._boundary_threshold else 0 for x in first_order_score_float])

        # 根据一阶导数划分区间
        boundary_state = 0
        score_segments = []
        score_segment = []
        for idx in range(first_order_score_binary.shape[0]):
            if first_order_score_binary[idx] == boundary_state:
                score_segment.append(self._scores[idx])
            if first_order_score_binary[idx] != boundary_state or idx == (first_order_score_binary.shape[0] - 1):
                if len(score_segment) >= 3:
                    score_segments.append(np.array(score_segment).mean())
                    score_segment = []

        # 判断是否为唤醒词
        if len(score_segments) < 2:
            return detection_bool

        find_low_threshold_bool = False
        find_high_threshold_bool = False
        for idy in range(len(score_segments)):
            if not find_low_threshold_bool and score_segments[idy] > self._detection_threshold_low:
                find_low_threshold_bool = True
            elif find_low_threshold_bool and score_segments[idy] > self._detection_threshold_high:
                find_high_threshold_bool = True
            elif find_low_threshold_bool and score_segments[idy] < self._detection_threshold_low:
                find_low_threshold_bool = False
            else:
                pass

            if find_low_threshold_bool and find_high_threshold_bool:
                detection_bool = True
                break
        # print(score_segments, detection_bool)
        return detection_bool
                

    def compute_conf(scores, word_num=2):
        scores = scores[:, 1:]
        h = np.zeros(scores.shape)
        M = scores.shape[1]
        Ts = scores.shape[0]
        # compute score for the first keyword
        h[0][0] = scores[0][0]
        for i in range(1, Ts):
            h[i][0] = max(h[i - 1][0], scores[i][0])
        # computing score for the remaining keywords
        for k in range(1, M):
            h[k][k] = h[k - 1][k - 1] * scores[k][k]
            for t in range(k + 1, Ts):
                h[t][k] = max(h[t - 1][k], h[t - 1][k - 1] * scores[t][k])

        return h[Ts - 1][M - 1] ** (1/word_num)


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
        self._response_time = 0

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

    @property
    def response_time(self):
        return self._response_time

    @response_time.setter
    def response_time(self, value):
        self._response_time = value


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
      _previous_top_label: Last founded command. Initial value is '_silence_'.
      _previous_top_time: The timestamp of _previous results. Default is -np.inf.
    """

    def __init__(self, labels, positove_lable_index, average_window_duration_ms, detection_threshold,
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
        if (self._previous_results.__len__() != 0 and
                current_time_ms < self._previous_results[0][0]):
            raise ValueError("Results must be fed in increasing time order, "
                             "but receive a timestamp of {}, which was earlier "
                             "than the previous one of {}".format(
                                 current_time_ms, self._previous_results[0][0]))

        # Add the latest result to the head of the deque.
        self._previous_results.append(
            [current_time_ms, latest_results[0], time.perf_counter()])

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
        average_scores = np.zeros(len(self._previous_results[0][1]))
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
            recognize_element.response_time = self._previous_results[-1][2] - \
                self._previous_results[0][2]
        else:
            recognize_element.is_new_command = False
            recognize_element.founded_command = SILENCE_LABEL


class RecognizeCommandsCountNumber(object):
    """Smooth the inference results by calculate the number of score greater than the threshold.

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
      _previous_top_label: Last founded command. Initial value is '_silence_'.
      _previous_top_time: The timestamp of _previous results. Default is -np.inf.
    """

    def __init__(self, labels, positove_lable_index, average_window_duration_ms, detection_threshold, detection_number_threshold,
                 suppression_ms, minimum_count):
        """Init the RecognizeCommands with parameters used for smoothing."""
        # Configuration
        self._labels = labels
        self._positove_lable_index = positove_lable_index
        self._average_window_duration_ms = average_window_duration_ms
        self._detection_threshold = detection_threshold
        self._detection_number_threshold = detection_number_threshold
        self._suppression_ms = suppression_ms
        self._minimum_count = minimum_count
        # Working Variable
        self._previous_results = collections.deque()
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
        if (self._previous_results.__len__() != 0 and
                current_time_ms < self._previous_results[0][0]):
            raise ValueError("Results must be fed in increasing time order, "
                             "but receive a timestamp of {}, which was earlier "
                             "than the previous one of {}".format(
                                 current_time_ms, self._previous_results[0][0]))

        # Add the latest result to the head of the deque.
        self._previous_results.append(
            [current_time_ms, latest_results[0], time.perf_counter()])

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

        # Calculate the number of score greater than the threshold.
        detection_number = 0
        for item in self._previous_results:
            score = item[1]
            if score[self._positove_lable_index] > self._detection_threshold:
                detection_number += 1
        recognize_element.score = detection_number

        time_since_last_top = current_time_ms - self._previous_top_time
        if self._previous_top_time == 0:
            time_since_last_top = np.inf

        if (recognize_element.score >= (self._detection_number_threshold * how_many_results) and
                time_since_last_top > self._suppression_ms):
            self._previous_top_time = current_time_ms
            recognize_element.is_new_command = True
            recognize_element.start_time = self._previous_results[0][0]
            recognize_element.end_time = self._previous_results[-1][0]
            recognize_element.founded_command = self._labels[self._positove_lable_index]
            recognize_element.response_time = self._previous_results[-1][2] - \
                self._previous_results[0][2]
        else:
            recognize_element.is_new_command = False
            recognize_element.founded_command = SILENCE_LABEL


class RecognizeCommandsAlign(object):
    """ 双边缘检测法检测输出结果，直接对未平滑结果进行处理，后续尝试平滑结果

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
      _previous_top_label: Last founded command. Initial value is '_silence_'.
      _previous_top_time: The timestamp of _previous results. Default is -np.inf.
    """

    def __init__(self, labels, positove_lable_index, average_window_duration_ms, detection_threshold_low, detection_threshold_high,
                 suppression_ms, minimum_count, align_type="transform", double_threshold_bool=True):
        """Init the RecognizeCommands with parameters used for smoothing."""
        # Configuration
        self._labels = labels
        self._positove_lable_index = positove_lable_index
        self._average_window_duration_ms = average_window_duration_ms
        self._detection_threshold = detection_threshold_high
        self._suppression_ms = suppression_ms
        self._minimum_count = minimum_count
        self._double_threshold_bool = double_threshold_bool
        # Working Variable
        self._previous_results = collections.deque()
        self._previous_top_time = 0
        self._align_type = align_type
        if self._double_threshold_bool:
            self._double_edge_detecting = DoubleEdgeDetecting(detection_threshold_low, detection_threshold_high)

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
        if (self._previous_results.__len__() != 0 and
                current_time_ms < self._previous_results[0][0]):
            raise ValueError("Results must be fed in increasing time order, "
                             "but receive a timestamp of {}, which was earlier "
                             "than the previous one of {}".format(
                                 current_time_ms, self._previous_results[0][0]))

        # Add the latest result to the head of the deque.
        self._previous_results.append(
            [current_time_ms, latest_results[0], time.perf_counter()])

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

        # 后处理操作，基于帧对齐模式，双边缘检测算法
        scores_list = []
        for item in self._previous_results:
            scores_list.append(item[1])

        if self._align_type == "transform":
            # 后处理方法：检测结果：unknow、小鱼、鱼小，检测边缘：小鱼、鱼小、小鱼，故将 3 维检测结果拼接为 4 维
            score_list_window = np.array(scores_list)
            score_list_window = np.concatenate(
                (score_list_window, score_list_window[:, 1].reshape(score_list_window.shape[0], 1)), axis=1)
            score = DoubleEdgeDetecting.compute_conf(
                score_list_window, word_num=3)
            recognize_element.score = score
        elif self._align_type == "word":
            # 后处理方法：检测结果：unknow、小、鱼，检测边缘：小、鱼、小、鱼，故将 3 维检测结果拼接为 5 维
            score_list_window = np.array(scores_list)
            score_list_window = np.concatenate(
                (score_list_window, score_list_window[:, 1:3].reshape(score_list_window.shape[0], 2)), axis=1)
            score = DoubleEdgeDetecting.compute_conf(
                score_list_window, word_num=4)
            recognize_element.score = score
        else:
            raise Exception(
                "[ERROR] Unknow align_type: {}, please check!".fomrat(align_type))

        time_since_last_top = current_time_ms - self._previous_top_time
        if self._previous_top_time == 0:
            time_since_last_top = np.inf

        if self._double_threshold_bool:
            find_bool = self._double_edge_detecting.process_latest_result(recognize_element.score, len(scores_list))
        else:
            find_bool = recognize_element.score >= self._detection_threshold
            
        if find_bool and time_since_last_top > self._suppression_ms:
            self._previous_top_time = current_time_ms
            recognize_element.is_new_command = True
            recognize_element.start_time = self._previous_results[0][0]
            recognize_element.end_time = self._previous_results[-1][0]
            recognize_element.founded_command = self._labels[self._positove_lable_index]
            recognize_element.response_time = self._previous_results[-1][2] - \
                self._previous_results[0][2]
        else:
            recognize_element.is_new_command = False
            recognize_element.founded_command = SILENCE_LABEL
