# openlogprobs

![openlogprobs on pypi](https://badge.fury.io/py/language-tool-python.svg)

![Test with PyTest](https://github.com/justinchiu/openlogprobs/workflows/Test%20with%20PyTest/badge.svg)

from openlogprobs import extract_logprobs
### topk search
extract_logprobs("gpt-3.5-turbo-instruct", "i like pie", topk=True)

### binary search
extract_logprobs("gpt-3.5-turbo-instruct", "i like pie", topk=False)


### Future work (help wanted!)

- support multiple logprobs (concurrent binary search)
- estimate costs for various APIs
- support checkpointing
