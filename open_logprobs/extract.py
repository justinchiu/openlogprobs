from tqdm import tqdm
import tiktoken
import numpy as np
from scipy.special import logsumexp
import os
import math

from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def topk(model, prefix, logit_bias=None, system=None):
    enc = tiktoken.encoding_for_model(model)
    if model == "gpt-3.5-turbo-instruct":
        if logit_bias is not None:
            response = client.completions.create(
                model=model,
                prompt=prefix,
                temperature=1,
                max_tokens=1,
                logit_bias=logit_bias,
                logprobs=5,
            )
        else:
            response = client.completions.create(
                model=model,
                prompt=prefix,
                temperature=1,
                max_tokens=1,
                logprobs=5,
            )
    else:
        raise NotImplementedError(f"Tried to get topk logprobs for: {model}")
    topk_dict = response.choices[0].logprobs.top_logprobs[0]
    return {enc.encode(x)[0]: y for x, y in topk_dict.items()}


def argmax(model, prefix, logit_bias=None, system=None):
    system = "You are a helpful assistant." if system is None else system

    enc = tiktoken.encoding_for_model(model)
    if model == "gpt-3.5-turbo-instruct":
        if logit_bias is not None:
            response = client.completions.create(
                model=model,
                prompt=prefix,
                temperature=0,
                max_tokens=1,
                logit_bias=logit_bias,
                n=1,
            )
        else:
            response = client.completions.create(
                model=model,
                prompt=prefix,
                temperature=0,
                max_tokens=1,
                n=1,
            )
        output = response.choices[0].text
        eos_idx = enc.encode(
            "<|endoftext|>", allowed_special={"<|endoftext|>", "<|im_start|>"}
        )[0]
        outputs = [choice.text for choice in response.choices]
    else:
        if logit_bias is not None:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prefix},
                ],
                temperature=0,
                max_tokens=1,
                logit_bias=logit_bias,
                n=1,
            )
        else:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prefix},
                ],
                temperature=0,
                max_tokens=1,
                n=1,
            )
        output = response.choices[0].message["content"]
        outputs = [choice.message["content"] for choice in response.choices]
        eos_idx = enc.encode(
            "<|endoftext|>", allowed_special={"<|endoftext|>", "<|im_start|>"}
        )[0]

    return enc.encode(output)[0]


def median_argmax(k, *args, **kwargs):
    return np.median([argmax(*args, **kwargs) for _ in range(k)])

def median_topk(k, *args, **kwargs):
    results = [topk(*args, **kwargs) for _ in range(k)]
    return {
        word: np.median([result[word] for result in results])
        for word in results[0].keys()
    }


def bisection_search(model, prefix, idx, k=5, low=0, high=32, eps=1e-8):
    # check if idx is the argmax
    num_calls = k
    if median_argmax(k, model, prefix) == idx:
        return 0, num_calls

    # initialize high
    logit_bias = {idx: high}
    while median_argmax(k, model, prefix, logit_bias) != idx:
        logit_bias[idx] *= 2
        num_calls += k
    high = logit_bias[idx]

    # improve estimate
    mid = (high + low) / 2
    while high >= low + eps:
        logit_bias[idx] = mid
        if median_argmax(k, model, prefix, logit_bias) == idx:
            high = mid
        else:
            low = mid
        mid = (high + low) / 2
        num_calls += k
    return -mid, num_calls


def topk_search(model, prefix, idx, k=5, high=40):
    # get raw topk, could be done outside and passed in
    topk_words = median_topk(k, model, prefix)
    highest_idx = list(topk_words.keys())[np.argmax(list(topk_words.values()))]
    if idx == highest_idx:
        return topk_words[idx], k
    num_calls = k

    # initialize high
    logit_bias = {idx: high}
    new_max_idx = median_argmax(k, model, prefix, logit_bias)
    num_calls += k
    while new_max_idx != idx:
        logit_bias[idx] *= 2
        new_max_idx = median_argmax(k, model, prefix, logit_bias)
        num_calls += k
    high = logit_bias[idx]

    output = median_topk(k, model, prefix, logit_bias)
    num_calls += k

    # compute normalizing constant
    diff = topk_words[highest_idx] - output[highest_idx]
    logZ = high - math.log(math.exp(diff) - 1)
    fv = output[idx] + math.log(math.exp(logZ) + math.exp(high)) - high
    logprob = fv - logZ

    return logprob, num_calls


class LockedOutput:
    def __init__(self, vocab_size, total_calls=0):
        self.lock = threading.Lock()
        self.total_calls = total_calls
        self.logits = np.zeros(vocab_size, dtype=np.float64)

    def add(self, calls, x, diff):
        with self.lock:
            self.total_calls += calls
            self.logits[x] = diff
            with open("temp/out.npy", "wb") as f:
                # TODO: better temp file saving
                np.save(f, self.logits)


def extract_logprobs(model, prefix, topk=False, eps=1e-6):
    enc = tiktoken.encoding_for_model(model)
    vocab_size = enc.n_vocab

    output = LockedOutput(vocab_size, total_calls=0)

    search = topk_search if topk else bisection_search

    def worker(x, output):
        logprob, num_calls = search(model, x, prefix)
        output.add(num_calls, x, logprob)

    with tqdm(total=vocab_size) as pbar:
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(worker, x, output) for x in range(vocab_size)]
            for future in as_completed(futures):
                pbar.update(1)

    return output.logits - logsumexp(outputs.logits), output.total_calls
