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

    assert False

def test_topk():
    assert False

def test_extract():
    assert False
