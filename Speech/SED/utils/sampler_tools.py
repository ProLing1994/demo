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

class BalancedTrainSampler(Base):
    def __init__(self, data_set, cfg):
        """Balanced sampler. Generate batch meta for training. Data are equally 
        sampled from different sound classes.
        """
        super(BalancedTrainSampler, self).__init__(data_set, cfg)
        
        # self.indexes = np.arange(self.audios_num)
            
        # # Shuffle indexes
        # self.random_state.shuffle(self.indexes)
        
        # self.pointer = 0
        
        self.samples_num_per_class = data_set.samples_num_per_class()

        # Training indexes of all sound classes. E.g.: 
        # [[0, 11, 12, ...], [3, 4, 15, 16, ...], [7, 8, ...], ...]
        self.indexes_per_class = data_set.indexes_per_class()
                    
        # Shuffle indexes
        for k in range(self.classes_num):
            self.random_state.shuffle(self.indexes_per_class[k])
        
        self.queue = []
        self.pointers_of_classes = [0] * self.classes_num

    def expand_queue(self, queue):
        classes_set = np.arange(self.classes_num).tolist()
        self.random_state.shuffle(classes_set)
        queue += classes_set
        return queue

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
                if len(self.queue) == 0:
                    self.queue = self.expand_queue(self.queue)

                class_id = self.queue.pop(0)
                pointer = self.pointers_of_classes[class_id]
                self.pointers_of_classes[class_id] += 1
                index = self.indexes_per_class[class_id][pointer]
                
                # When finish one epoch of a sound class, then shuffle its indexes and reset pointer
                if self.pointers_of_classes[class_id] >= self.samples_num_per_class[class_id]:
                    self.pointers_of_classes[class_id] = 0
                    self.random_state.shuffle(self.indexes_per_class[class_id])

                batch_meta.append(index)
                i += 1

            yield batch_meta

    def state_dict(self):
        state = {'indexes_per_class': self.indexes_per_class, 
                'queue': self.queue, 
                'pointers_of_classes': self.pointers_of_classes}
        return state
            
    def load_state_dict(self, state):
        self.indexes_per_class = state['indexes_per_class']
        self.queue = state['queue']
        self.pointers_of_classes = state['pointers_of_classes']


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