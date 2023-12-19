# openlogprobs

#### 🪄 openlogprobs is a Python API for extracting log-probabilities from language model APIs 🪄 </p>


```bash
pip install openlogprobs
```

<hr>

![openlogprobs on pypi](https://badge.fury.io/py/openlogprobs.svg)

![Test with PyTest](https://github.com/justinchiu/openlogprobs/workflows/Test%20with%20PyTest/badge.svg)

Many API-based language model services hide the log-probability outputs from their models. One reason is security – language model outputs can reveal information about their inputs and can be used for efficient model distillation. Another reason is a practical one: serving 30,000 (or whatever the vocabulary size is) floats via an API would take way too much data for a typical API request. So this information is hidden to you.

However, most APIs also allow a 'logit bias' argument to positively or negatively influence the likelihood of certain tokens in language model output. It turns out, though, that we can use this logit bias on individual tokens to reverse-engineer their log probabilities. We developed an algorithm to do this efficiently, which effectively allows us to extract *full probability vectors* via APIs such as the OpenAI API. For more information, read the below section about the algorithm, or read the code in openlogprobs/extract.py.


## Usage

### topk search

If the API exposes the top-k log-probabilities, we can efficiently extract the next-token probabilities via our 'topk' algorithm:

```python
from openlogprobs import extract_logprobs
extract_logprobs("gpt-3.5-turbo-instruct", "i like pie", topk=True)
```

### binary search

If the API does not expose top-k logprobs, we can still extract the distribution, but it takes more language model calls:

```python
from openlogprobs import extract_logprobs
extract_logprobs("gpt-3.5-turbo-instruct", "i like pie", topk=False)
```

### Future work (help wanted!)

- support multiple logprobs (concurrent binary search)
- estimate costs for various APIs
- support checkpointing

## Algorithm

Our algorithm is esssentially a binary search (technically 'univariate bisection' on a continuous variable) where we apply different amounts of logit bias to make certain tokens likely enough to appear in the generation. This allows us to estimate the probability of any token relative to the most likely token. To obtain the full vector of probabilities, we can run this binary search on every token in the vocabulary. Note that essentially all models support logit bias, and for that to work, all models that support logit bias must be open-vocabulary.

Here's a crude visualization of how our algorithm works for a single token:

<img src="https://github.com/justinchiu/openlogprobs/raw/main/vis.png" width="600"/>

Each API call (purple) brings us successively closer to the true token probability (green).


## Language Model Inversion paper

This algorithm was developed mainly by Justin Chiu to facilitate the paper [*Language Model Inversion*](https://arxiv.org/abs/2311.13647). If you're using our algorithm in academic research, please cite our paper:

```
@misc{morris2023language,
      title={Language Model Inversion}, 
      author={John X. Morris and Wenting Zhao and Justin T. Chiu and Vitaly Shmatikov and Alexander M. Rush},
      year={2023},
      eprint={2311.13647},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
