import numpy as np
import threading
from itertools import islice
import sys

if sys.version_info.minor < 12:

    def batched(iterable, n):
        """From https://docs.python.org/3.11/library/itertools.html#itertools-recipes"""
        "Batch data into tuples of length n. The last batch may be shorter."
        # batched('ABCDEFG', 3) --> ABC DEF G
        if n < 1:
            raise ValueError("n must be at least one")
        it = iter(iterable)
        while batch := tuple(islice(it, n)):
            yield batch

else:
    from itertools import batched


class LockedOutput:
    def __init__(self, vocab_size, total_calls=0):
        self.lock = threading.Lock()
        self.total_calls = total_calls
        self.logits = np.zeros(vocab_size, dtype=np.float64)

    def add(self, calls, x, diff):
        with self.lock:
            self.total_calls += calls
            self.logits[x] = diff
