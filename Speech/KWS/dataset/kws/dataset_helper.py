  
SILENCE_LABEL = '_silence_'
SILENCE_INDEX = 0
UNKNOWN_WORD_LABEL = '_unknown_'
UNKNOWN_WORD_INDEX = 1
BACKGROUND_NOISE_DIR_NAME = '_background_noise_'
  
def load_label_index(positive_label):
  # data index
  label_index = {}
  for index, positive_word in enumerate(positive_label):
    label_index[positive_word] = index + 2
  label_index.update({SILENCE_LABEL: SILENCE_INDEX, UNKNOWN_WORD_LABEL: UNKNOWN_WORD_INDEX})
  return label_index