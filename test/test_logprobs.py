from typing import Dict

import numpy as np
import pytest
from scipy.special import log_softmax
import transformers

from openlogprobs import (
    extract_logprobs,
    # OpenAIModel,
)
from openlogprobs.extract import (
    bisection_search,
    topk_search,
)
from openlogprobs.models import Model

prefix = "Should i take this class or not? The professor of this class is not good at all. He doesn't teach well and he is always late for class."


def load_fake_logits(vocab_size: int) -> np.ndarray:
    np.random.seed(42)
    logits = np.random.randn(vocab_size)
    logits[1] += 10
    logits[12] += 20
    logits[13] += 30
    logits[24] += 30
    logits[35] += 30
    return logits


class FakeModel(Model):
    """Represents a fake API with a temperature of 1. Used for testing."""

    def __init__(self, vocab_size: int = 100, get_logits=None):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
        self.fake_vocab_size = vocab_size
        if get_logits is None:
            self.logits = load_fake_logits(self.vocab_size)[:vocab_size]
        else:
            self.logits = get_logits(vocab_size)

    @property
    def vocab_size(self):
        return self.fake_vocab_size

    def _idx_to_str(self, idx: int) -> str:
        return self.tokenizer.decode([idx], skip_special_tokens=True)

    def _add_logit_bias(self, logit_bias: Dict[str, float]) -> np.ndarray:
        logits = self.logits.copy()
        for token_idx, bias in logit_bias.items():
            logits[token_idx] += bias
        logits = logits.astype(np.double)
        return log_softmax(logits)

    def argmax(self, prefix: str, logit_bias: Dict[str, float] = {}) -> int:
        logits = self._add_logit_bias(logit_bias)
        return logits.argmax()

    def topk(self, prefix: str, logit_bias: Dict[str, float] = {}) -> Dict[int, float]:
        k = 5  # TODO: what topk?
        logits = self._add_logit_bias(logit_bias)
        topk = logits.argsort()[-k:]
        return {k: logits[k] for k in topk}


@pytest.fixture
def model():
    # return OpenAIModel("gpt-3.5-turbo-instruct")
    return FakeModel()


@pytest.fixture
def uniform_model():
    # return OpenAIModel("gpt-3.5-turbo-instruct")
    return FakeModel(get_logits=np.ones)


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
    for _trial in range(10):
        estimated_probs = {
            word: topk_search(model, prefix, word) for word in topk_words.keys()
        }
        estimated_probs = np.array(sorted([x[0] for x in estimated_probs.values()]))
        probs.append(estimated_probs)
    probs = np.stack(probs)
    assert np.allclose(true_probs, np.median(probs, 0), atol=1e-5)


def test_extract_topk(model):
    true_logprobs = log_softmax(model.logits)
    extracted_logprobs, num_calls = extract_logprobs(
        model, prefix="test", method="topk", multithread=False, k=1
    )
    np.testing.assert_allclose(true_logprobs, extracted_logprobs)
    assert num_calls == 298


def test_extract_bisection(model):
    true_logprobs = log_softmax(model.logits)
    extracted_logprobs, num_calls = extract_logprobs(
        model, prefix="test", method="bisection", multithread=False, k=1
    )
    np.testing.assert_allclose(true_logprobs, extracted_logprobs)
    assert num_calls == 3270


def test_extract_exact(model):
    true_logprobs = log_softmax(model.logits)
    extracted_logprobs, num_calls = extract_logprobs(
        model, prefix="test", method="exact", multithread=False
    )
    np.testing.assert_allclose(true_logprobs, extracted_logprobs)
    assert num_calls < len(true_logprobs)


def test_extract_exact_parallel(model):
    true_logprobs = log_softmax(model.logits)
    extracted_logprobs, num_calls = extract_logprobs(
        model,
        prefix="test",
        method="exact",
        multithread=False,
        parallel=True,
    )
    np.testing.assert_allclose(true_logprobs, extracted_logprobs)
    assert num_calls < len(true_logprobs)


def test_extract_topk_multithread(model):
    true_logprobs = log_softmax(model.logits)
    extracted_logprobs, num_calls = extract_logprobs(
        model, prefix="test", method="topk", multithread=True, k=1
    )
    np.testing.assert_allclose(true_logprobs, extracted_logprobs)
    assert num_calls == 298


def test_extract_exact_multithread(model):
    true_logprobs = log_softmax(model.logits)
    extracted_logprobs, num_calls = extract_logprobs(
        model, prefix="test", method="exact", multithread=True
    )
    np.testing.assert_allclose(true_logprobs, extracted_logprobs)
    assert num_calls < len(true_logprobs)


def test_extract_exact_parallel_multithread(model):
    true_logprobs = log_softmax(model.logits)
    extracted_logprobs, num_calls = extract_logprobs(
        model, prefix="test", method="exact", multithread=True, parallel=True
    )
    np.testing.assert_allclose(true_logprobs, extracted_logprobs)
    assert num_calls < len(true_logprobs)


def test_extract_exact_parallel_multithread_uniform(uniform_model):
    true_logprobs = log_softmax(uniform_model.logits)
    extracted_logprobs, num_calls = extract_logprobs(
        uniform_model,
        prefix="test",
        method="exact",
        parallel=True,
    )
    np.testing.assert_allclose(true_logprobs, extracted_logprobs)
    assert num_calls < len(true_logprobs)
