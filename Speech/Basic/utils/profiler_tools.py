from time import perf_counter as timer
from collections import OrderedDict
import numpy as np


class Profiler:
    def __init__(self, summarize_every=5, disabled=False, logger=None):
        self.last_tick = timer()
        self.logs = OrderedDict()
        self.summarize_every = summarize_every
        self.disabled = disabled
        self.logger = logger
    
    def tick(self, name):
        if self.disabled:
            return
        
        # Log the time needed to execute that function
        if not name in self.logs:
            self.logs[name] = []
        if len(self.logs[name]) >= self.summarize_every:
            self.summarize()
            if self.logger is not None:
                self.summarize_logger()
            self.purge_logs()
        self.logs[name].append(timer() - self.last_tick)
        
        self.reset_timer()
        
    def purge_logs(self):
        for name in self.logs:
            self.logs[name].clear()
    
    def reset_timer(self):
        self.last_tick = timer()
    
    def summarize(self):
        n = max(map(len, self.logs.values()))
        assert n == self.summarize_every
        print("\nAverage execution time over %d steps:" % n)

        name_msgs = ["%s (%d/%d):" % (name, len(deltas), n) for name, deltas in self.logs.items()]
        pad = max(map(len, name_msgs))
        for name_msg, deltas in zip(name_msgs, self.logs.values()):
            print("  %s  mean: %4.0fms   std: %4.0fms" % 
                  (name_msg.ljust(pad), np.mean(deltas) * 1000, np.std(deltas) * 1000))
        print("", flush=True)    
    
    def summarize_logger(self):
        n = max(map(len, self.logs.values()))

        msg = 'steps: {}'.format(n)
        for name, deltas in self.logs.items():
            msg += ', {}: {:0.2f}/{:0.2f}'.format(name, np.mean(deltas) * 1000, np.std(deltas) * 1000)
        self.logger.info(msg)