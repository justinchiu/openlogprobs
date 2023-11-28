import numpy as np
from open_logprobs.extract import (
    extract_logprobs,
    bisection_search,
    topk_search,
    topk as topk_,
)

prefix = "Should i take this class or not? The professor of this class is not good at all. He doesn't teach well and he is always late for class."
model = "gpt-3.5-turbo-instruct"
topk_words = topk_(model, prefix)


def test_bisection():
    true_sorted_logprobs = np.array(sorted(topk_words.values()))
    true_diffs = true_sorted_logprobs - true_sorted_logprobs.max()

    estimated_diffs = {word: bisection_search(model, prefix, word) for word in topk_words.keys()}
    estimated_diffs = np.array(sorted([x[0] for x in estimated_diffs.values()]))
    assert np.allclose(true_diffs, estimated_diffs, atol=1e-5)

def test_topk():
    true_probs = np.array(sorted(topk_words.values()))

    estimated_probs = {word: topk_search(model, prefix, word) for word in topk_words.keys()}
    estimated_probs = np.array(sorted([x[0] for x in estimated_probs.values()]))
    assert np.allclose(true_probs, estimated_probs, atol=1e-5)

def test_extract():
    assert False
