import tqdm
import tiktoken
import numpy as np
from scipy.special import logsumexp
import math
from functools import partial, reduce
from operator import or_ as union
from typing import Literal

from concurrent.futures import ThreadPoolExecutor

from openlogprobs.models import Model
from openlogprobs.utils import batched


def exact_solve(
    model: Model,
    prefix: str,
    idx: list[int],
    bias: float = 5.0,
    top_logprob: float | None = None,
) -> tuple[dict[int, float], set[int], int]:
    """Parallel exact solve based on https://mattf1n.github.io/openlogprobs.html"""
    logit_bias = {i: bias for i in idx}
    topk_words = model.topk(prefix, logit_bias)
    if all(i in topk_words for i in idx):
        biased_logprobs = np.array([topk_words[i] for i in idx])
        log_biased_prob = logsumexp(biased_logprobs)
        logprobs = biased_logprobs - np.logaddexp(
            bias + np.log1p(-np.exp(log_biased_prob)), log_biased_prob
        )
        return dict(zip(idx, logprobs)), set(), 1
    else:
        if top_logprob is None:
            missing_tokens = set(idx) - set(topk_words)
            raise TypeError(
                f"Tokens {missing_tokens} not in top-k with bias {bias}."
                "Either increase bias or provide top unbiased logprob (top_logprob)"
            )
        success_idxs = list(i for i in idx if i in topk_words)
        fail_idxs = set(idx) - set(topk_words)
        biased_top_logprob = max(
            logprob for i, logprob in topk_words.items() if i not in idx
        )
        biased_logprobs = np.array([topk_words[i] for i in success_idxs])
        logprobs = biased_logprobs - biased_top_logprob + top_logprob - bias
        return dict(zip(success_idxs, logprobs)), fail_idxs, 1


def bisection_search(
    model: Model, prefix: str, idx: int, k=1, low=0, high=32, eps=1e-8
):
    # check if idx is the argmax
    num_calls = k
    if model.argmax(prefix) == idx:
        return 0, num_calls

    # initialize high
    logit_bias = {idx: high}
    while model.argmax(prefix, logit_bias) != idx:
        logit_bias[idx] *= 2
        num_calls += k
    high = logit_bias[idx]

    # improve estimate
    mid = (high + low) / 2
    while high >= low + eps:
        logit_bias[idx] = mid
        if model.argmax(prefix, logit_bias) == idx:
            high = mid
        else:
            low = mid
        mid = (high + low) / 2
        num_calls += k
    return -mid, num_calls


def topk_search(model: Model, prefix: str, idx: int, k=1, high=40):
    # get raw topk, could be done outside and passed in
    topk_words = model.topk(prefix)
    highest_idx = list(topk_words.keys())[np.argmax(list(topk_words.values()))]
    if idx == highest_idx:
        return topk_words[idx], k
    num_calls = k

    # initialize high
    logit_bias = {idx: high}
    new_max_idx = model.argmax(prefix, logit_bias)
    num_calls += k
    while new_max_idx != idx:
        logit_bias[idx] *= 2
        new_max_idx = model.argmax(prefix, logit_bias)
        num_calls += k
    high = logit_bias[idx]

    output = model.topk(prefix, logit_bias)
    num_calls += k

    # compute normalizing constant
    diff = topk_words[highest_idx] - output[highest_idx]
    logZ = high - math.log(math.exp(diff) - 1)
    fv = output[idx] + math.log(math.exp(logZ) + math.exp(high)) - high
    logprob = fv - logZ

    return logprob, num_calls


def extract_logprobs(
    model: Model,
    prefix: str,
    method: Literal["bisection", "topk", "exact"] = "bisection",
    k: int = 5,
    eps: float = 1e-6,
    multithread: bool = False,
    bias: float = 5.0,
    parallel: bool = False,
):
    vocab_size = model.vocab_size

    if method == "exact":
        logprob_dict = model.topk(prefix)
        top_logprob = max(logprob_dict.values())
        bias += top_logprob - min(logprob_dict.values())
        remaining = set(range(vocab_size)) - set(logprob_dict)
        total_calls = 0
        if multithread:
            executor = ThreadPoolExecutor(max_workers=8)
            map_func = executor.map
        else:
            map_func = map
        while remaining:
            search_results = map_func(
                partial(
                    exact_solve,
                    model,
                    prefix,
                    bias=bias,
                    top_logprob=top_logprob,
                ),
                batched(remaining, k),
            )
            logprob_dicts, skipped, calls = zip(*search_results)
            logprob_dict |= reduce(union, logprob_dicts)
            remaining = set.union(*skipped)
            total_calls += sum(calls)
            bias += 5
        if multithread:
            executor.shutdown()
        logprobs = np.array([logprob_dict[i] for i in range(vocab_size)])
        return logprobs, total_calls
    else:
        search_func = topk_search if method == "topk" else bisection_search
        search = partial(search_func, model, prefix, k=k)
        vocab = list(range(vocab_size))
        if multithread:
            with ThreadPoolExecutor(max_workers=8) as executor:
                search_results = executor.map(search, tqdm.tqdm(vocab))
        else:
            search_results = map(search, tqdm.tqdm(vocab))
        logit_list, calls = zip(*search_results)
        logits = np.array(logit_list)
        return logits - logsumexp(logits), sum(calls)
