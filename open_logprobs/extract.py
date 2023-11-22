from tqdm import tqdm
import openai
import tiktoken
import numpy as np
from scipy.special import logsumexp


def topk(model, prefix, logit_bias=None, system=None):
    enc = tiktoken.encoding_for_model(model)
    if model == "gpt-3.5-turbo-instruct":
        if logit_bias is not None:
            response = openai.Completion.create(
                model=model,
                prompt=prefix,
                temperature=1,
                max_tokens=1,
                logit_bias=logit_bias,
                logprobs=5,
            )
        else:
            response = openai.Completion.create(
                model=model,
                prompt=prefix,
                temperature=1,
                max_tokens=1,
                logprobs=5,
            )
    else:
        raise NotImplementedError(f"Tried to get topk logprobs for: {model}")
    topk_dict = response.choices[0].logprobs.top_logprobs[0]
    return {
        enc.encode(x)[0]: y
        for x,y in topk_dict.items()
    }

def argmax(model, prefix, logit_bias=None, system=None):
    system = "You are a helpful assistant." if system is None else system

    enc = tiktoken.encoding_for_model(model)
    if model == "gpt-3.5-turbo-instruct":
        if logit_bias is not None:
            response = openai.Completion.create(
                model=model,
                prompt=prefix,
                temperature=0,
                max_tokens=1,
                logit_bias=logit_bias,
                n=1,
            )
        else:
            response = openai.Completion.create(
                model=model,
                prompt=prefix,
                temperature=0,
                max_tokens=1,
                n=1,
            )
        output = response.choices[0].text
        eos_idx = enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>", "<|im_start|>"})[0]
        outputs = [choice.text for choice in response.choices]
    else:
        if logit_bias is not None:
            response = openai.ChatCompletion.create(
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
            response = openai.ChatCompletion.create(
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
        eos_idx = enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>", "<|im_start|>"})[0]

    # just give eos_idx if there's a weird failure
    # TODO: fix this for the weird vocabs
    if response.choices[0].finish_reason == "length":
        idx = enc.encode(output)[0] if output else eos_idx
    elif response.choices[0].finish_reason == "stop":
        idx = eos_idx
    else:
        import pdb; pdb.set_trace()

    return idx


def bisection_search(model, idx, prefix, low=0, high=32, eps=1e-8):
    # initialize high
    logit_bias = {idx: high}
    num_calls = 1
    while argmax(model, prefix, logit_bias) != idx:
        logit_bias[idx] *= 2
        num_calls += 1
    high = logit_bias[idx]

    # improve estimate
    mid = (high + low) / 2
    while high > low + eps:
        logit_bias[idx] = mid
        if argmax(model, prefix, logit_bias) == idx:
            high = mid
        else:
            low = mid
        mid = (high + low) / 2
        num_calls += 1
    return -mid, num_calls

def prob_search(model, idx, prefix, high=40):
    # get raw topk, could be done outside and passed in
    topk = topk(model, prefix)
    highest_idx = list(topk.keys())[np.argmax(list(topk.values()))]
    if idx == highest_idx:
        return topk[idx], 1

    # initialize high
    logit_bias = {idx: high}
    new_max_idx = sampler.sample(prefix, 1, logit_bias, temperature=0).argmax
    num_calls = 2
    while new_max_idx != idx:
        logit_bias[idx] *= 2
        new_max_idx = sampler.sample(prefix, 1, logit_bias, temperature=0).argmax
        num_calls += 1
    high = logit_bias[idx]
    output = sampler.topk(prefix, logit_bias)
    num_calls += 1

    # compute normalizing constant
    diff = topk[highest_idx] - output[highest_idx]
    logZ = high - math.log(math.exp(diff) - 1)
    # ideally would be output[idx], but it seems like openai sometimes returns weird things?
    fv = np.max(list(output.values())) + math.log(math.exp(logZ) + math.exp(high)) - high
    logprob = fv - logZ

    if np.max(list(output.values())) == output[highest_idx]:
        # highest probability word didnt change
        print("MESSED UP", idx, high, new_max_idx, highest_idx, topk, output)
        import pdb; pdb.set_trace()

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


def extract(model, prefix, topk=False, eps=1e-6):
    vocab_size = sampler.vocab_size
    logit_bias = {}

    output = LockedOutput(vocab_size, total_calls = 0)

    search = prob_search if topk else bisection_search

    def worker(x, output):
        # TODO test if variable capture works
        logprob, num_calls = search(model, x, prefix)
        output.add(num_calls, x, logprob)

    with tqdm(total=vocab_size) as pbar:
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(worker, x, output) for x in range(vocab_size)]
            for future in as_completed(futures):
                pbar.update(1)

    return output.logits - logsumexp(outputs.logits), output.total_calls

