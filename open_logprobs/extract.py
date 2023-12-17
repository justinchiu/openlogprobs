from tqdm import tqdm
import tiktoken
import numpy as np
from scipy.special import logsumexp
import math

from concurrent.futures import ThreadPoolExecutor, as_completed

from open_logprobs.models import Model
from open_logprobs.utils import LockedOutput



def bisection_search(model: Model, prefix: str, idx: int, k=5, low=0, high=32, eps=1e-8):
    # check if idx is the argmax
    num_calls = k
    if model.median_argmax(k, model, prefix) == idx:
        return 0, num_calls

    # initialize high
    logit_bias = {idx: high}
    while model.median_argmax(k, model, prefix, logit_bias) != idx:
        logit_bias[idx] *= 2
        num_calls += k
    high = logit_bias[idx]

    # improve estimate
    mid = (high + low) / 2
    while high >= low + eps:
        logit_bias[idx] = mid
        if model.median_argmax(k, model, prefix, logit_bias) == idx:
            high = mid
        else:
            low = mid
        mid = (high + low) / 2
        num_calls += k
    return -mid, num_calls


def topk_search(model: Model, prefix: str, idx: int, k=5, high=40):
    # get raw topk, could be done outside and passed in
    topk_words = model.median_topk(k, model, prefix)
    highest_idx = list(topk_words.keys())[np.argmax(list(topk_words.values()))]
    if idx == highest_idx:
        return topk_words[idx], k
    num_calls = k

    # initialize high
    logit_bias = {idx: high}
    new_max_idx = model.median_argmax(k, model, prefix, logit_bias)
    num_calls += k
    while new_max_idx != idx:
        logit_bias[idx] *= 2
        new_max_idx = model.median_argmax(k, model, prefix, logit_bias)
        num_calls += k
    high = logit_bias[idx]

    output = model.median_topk(k, model, prefix, logit_bias)
    num_calls += k

    # compute normalizing constant
    diff = topk_words[highest_idx] - output[highest_idx]
    logZ = high - math.log(math.exp(diff) - 1)
    fv = output[idx] + math.log(math.exp(logZ) + math.exp(high)) - high
    logprob = fv - logZ

    return logprob, num_calls


def extract_logprobs(model: Model, prefix: str, topk=False, k=5, eps=1e-6):
    vocab_size = model.vocab_size
    output = LockedOutput(vocab_size, total_calls=0)

    search = topk_search if topk else bisection_search

    def worker(x, output):
        logprob, num_calls = search(model, x, prefix, k=k)
        output.add(num_calls, x, logprob)

    with tqdm(total=vocab_size) as pbar:
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(worker, x, output) for x in range(vocab_size)]
            for future in as_completed(futures):
                pbar.update(1)

    return output.logits - logsumexp(output.logits), output.total_calls
