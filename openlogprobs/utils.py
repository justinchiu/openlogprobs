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
