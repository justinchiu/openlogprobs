from typing import Dict

import numpy as np
import pytest
from scipy.special import log_softmax
import transformers

from open_logprobs import (
    extract_logprobs,
    OpenAIModel,
)
from open_logprobs.extract import (
    bisection_search,
    topk_search,
)
from open_logprobs.models import Model

prefix = "Should i take this class or not? The professor of this class is not good at all. He doesn't teach well and he is always late for class."


def load_fake_logits(vocab_size: int) -> np.ndarray:
    logits = np.random.randn(vocab_size)
    logits[1] += 10
    logits[12] += 20
    logits[13] += 30
    logits[24] += 30
    logits[35] += 30
    return log_softmax(logits)


class FakeModel(Model):
    def __init__(self):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
        self.logits = load_fake_logits(self.vocab_size)

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    def _idx_to_str(self, idx: int) -> str:
        return self.tokenizer.decode([idx], skip_special_tokens=True)

    def _add_logit_bias(self, logit_bias: Dict[str, float]) -> np.ndarray:
        print("_add_logit_bias", logit_bias)
        logits = self.logits.copy()
        for token, bias in logit_bias.items():
            token_idx = self.tokenizer.vocab[token]
            logits[token_idx] += bias
        logits = logits.astype(np.double)
        logits = np.exp(logits)
        return log_softmax(logits)
    
    def argmax(self, prefix: str, logit_bias: Dict[str, float] = {}) -> str:
        logits = self._add_logit_bias(logit_bias)
        argmax = logits.argmax()
        return self._idx_to_str(argmax)
    
    def topk(self, prefix: str, logit_bias: Dict[str, float] = {}) -> Dict[str, float]:
        k = 5 # TODO: what topk?
        logits = self._add_logit_bias(logit_bias)
        topk = self.logits.argsort()[-k:]
        return {self._idx_to_str(k): logits[k] for k in topk}


@pytest.fixture
def model():
    # return OpenAIModel("gpt-3.5-turbo-instruct")
    return FakeModel()

@pytest.fixture
def topk_words(model):
    return model.topk(prefix)

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