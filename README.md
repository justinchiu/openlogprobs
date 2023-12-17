# open_logprobs

from open_logprobs import extract_logprobs
### topk search
extract_logprobs("gpt-3.5-turbo-instruct", "i like pie", topk=True)

### binary search
extract_logprobs("gpt-3.5-turbo-instruct", "i like pie", topk=False)


[todo]
[ ] refactor
[ ] add fixed output model
[ ] test fixed output model
[ ] support multiple logprobs
[ ] add linting 
[ ] run tests on github
[ ] write readme
[ ] add cost estimates subsection to readme
[ ] tweet




[tweet]
New Github library: open_logprobs

Language models give us the probability of the next token. However,
OpenAI hides this information, and either tells you a single token
or the probabilities of the top 5 (discarding the other 31,000 in the vocabulary)

We developed algorithms for recovering the *full* probability vector by making many calls to openAI. Our algorithms work by taking advantage of the logit_bias parameter and doing a binary search to compare the probabilities of individual tokens to those of the top token

This is useful for:
- distillation
- training data detection (weijia's paper: ...)
- recovering prompts (our paper on language model inversion)
- interpretability

Link here:


(then do a follow-up tweet about justin)