import collections
import sys
import time 

from wakeup.dataset_helper import *


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
        self._previous_results.append([current_time_ms, latest_results[0], time.perf_counter()])

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
            recognize_element.response_time = self._previous_results[-1][2] - self._previous_results[0][2]
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
      _label_count: The length of label list.
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
        self._previous_results.append([current_time_ms, latest_results[0], time.perf_counter()])

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
            recognize_element.response_time = self._previous_results[-1][2] - self._previous_results[0][2]
        else:
            recognize_element.is_new_command = False        
            recognize_element.founded_command = SILENCE_LABEL