import numpy as np
import threading

class LockedOutput:
    def __init__(self, vocab_size, total_calls=0):
        self.lock = threading.Lock()
        self.total_calls = total_calls
        self.logits = np.zeros(vocab_size, dtype=np.float64)

    def add(self, calls, x, diff):
        with self.lock:
            self.total_calls += calls
            self.logits[x] = diff