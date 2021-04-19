import numpy as np


class Base(object):
    def __init__(self, data_set, cfg):
        """
        Base class of train sampler.
        """
        self.batch_size = cfg.train.batch_size
        self.random_state = np.random.RandomState(cfg.debug.seed)
        
        self.audios_num = len(data_set)
        self.classes_num = cfg.dataset.label.num_classes


class TrainSampler(Base):
    def __init__(self, data_set, cfg):
        """
        Sampler. Generate batch meta for training.
        """
        super(TrainSampler, self).__init__(data_set, cfg)
        
        self.indexes = np.arange(self.audios_num)
            
        # Shuffle indexes
        self.random_state.shuffle(self.indexes)
        
        self.pointer = 0

    def __iter__(self):
        """Generate batch meta for training. 
        
        Returns:
            batch_meta: e.g.: [index, ...]
        """
        batch_size = self.batch_size

        while True:
            batch_meta = []
            i = 0
            while i < batch_size:
                index = self.indexes[self.pointer]
                self.pointer += 1

                # Shuffle indexes and reset pointer
                if self.pointer >= self.audios_num:
                    self.pointer = 0
                    self.random_state.shuffle(self.indexes)
                
                batch_meta.append(index)
                i += 1

            yield batch_meta

    def state_dict(self):
        state = {
            'indexes': self.indexes,
            'pointer': self.pointer}
        return state
            
    def load_state_dict(self, state):
        self.indexes = state['indexes']
        self.pointer = state['pointer']


class EvaluateSampler(object):
    def __init__(self, data_set, cfg):
        """
        Evaluate sampler. Generate batch meta for evaluation.
        """
        # self.batch_size = cfg.train.batch_size
        self.batch_size = 16
        self.audios_num = len(data_set)

    def __iter__(self):
        """Generate batch meta for training. 
        
        Returns:
            batch_meta: e.g.: [index, ...]
        """
        batch_size = self.batch_size
        pointer = 0

        while pointer < self.audios_num:
            batch_indexes = np.arange(pointer, min(pointer + batch_size, self.audios_num))

            batch_meta = []

            for index in batch_indexes:
                batch_meta.append(index)

            pointer += batch_size
            yield batch_meta