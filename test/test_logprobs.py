import numpy as np
import pytest

from open_logprobs import (
    extract_logprobs,
    bisection_search,
    topk_search,
    topk as topk_,
    Model,
    OpenAIModel
)

prefix = "Should i take this class or not? The professor of this class is not good at all. He doesn't teach well and he is always late for class."

class FakeModel(Model):
    def __init__(self):
        self.logits = load_logits(??)...
        self.tokenizer = ??

    @property
    def vocab_size(self):
        return self.logits.numel()

    def _add_logit_bias(logit_bias: Dict[str, float]) -> np.ndarray:
        logits = self.logits.clone()
        for token, bias in logit_bias:
            token_idx = self.tokenizer.vocab[token]
            logits[token_idx] += bias
        return logits
    
    def argmax(self, prefix: str, logit_bias: Dict[str, float] = {}) -> str
        logits = self._add_logit_bias(logit_bias)
        return self.logits.argmax()
    
    def topk(self, prefix: str, logit_bias: Dict[str, float] = {}) -> Dict[str, float]:
        logits = self._add_logit_bias(logit_bias)
        return self.logits.topk(??)


@pytest.fixture
def model():
    # return OpenAIModel("gpt-3.5-turbo-instruct")
    return FakeModel()

@pytest.fixture
def topk_words(model):
    return topk_(model, prefix)

def test_bisection(model, topk_words):
    true_sorted_logprobs = np.array(sorted(topk_words.values()))
    true_diffs = true_sorted_logprobs - true_sorted_logprobs.max()

    estimated_diffs = {
        word: bisection_search(model, prefix, word) for word in topk_words.keys()
    }
    estimated_diffs = np.array(sorted([x[0] for x in estimated_diffs.values()]))
    assert np.allclose(true_diffs, estimated_diffs, atol=1e-5)


def test_topk(model, topk_words):
    true_probs = np.array(sorted(topk_words.values()))

    estimated_probs = {
        word: topk_search(model, prefix, word) for word in topk_words.keys()
    }
    estimated_probs = np.array(sorted([x[0] for x in estimated_probs.values()]))
    assert np.allclose(true_probs, estimated_probs, atol=1e-5)


def test_topk_consistency(model, topk_words):
    true_probs = np.array(sorted(topk_words.values()))

    probs = []
    for trial in range(10):
        estimated_probs = {
            word: topk_search(model, prefix, word) for word in topk_words.keys()
        }
        estimated_probs = np.array(sorted([x[0] for x in estimated_probs.values()]))
        probs.append(estimated_probs)
    probs = np.stack(probs)
    assert np.allclose(true_probs, np.median(probs, 0), atol=1e-5)


def test_extract():
    assert False