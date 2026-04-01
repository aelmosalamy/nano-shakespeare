# nano-shakespeare

`nano-shakespeare` built based on Andrej Karpathy's ![Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY).

I tried to add comments to most of the information he called out. Even with 0 knowlege about ML (never used `pytorch`
before, this exercise helped me form an appreciation of the engineering that goes into it. I was also able to formulate
an understanding of the kind of decisions and areas of knowledge that would help you build and design systems in this
domain.

Now, how does this thing differ from ChatGPT?

1. We need a pre-training phase (we did a baby version of that)
2. We need a fine-tuning phase

First, in OpenAI's pre-training stage:

- ChatGPT uses different vocabularly, sub-word units as tokens, we used character-level tokens. Our Shakespeare dataset
  is a ~1M tokens in our vocabularly, would be around 300K tokens in theirs.
- GPT-3 from their paper "Language Models are a Few Shot Learners" had 175B parameters, our is ~10M parameters
- GPT-3 had 96 layers, 12288 n_embed, 96 heads, 128 head_size, with a batch_size of 3.2M, learning_rate 0.6*1e-4
- Obviously, their dataset covers almost the entirety of the internet. Trained on 300 billion tokens (a 1e6 multiplier)

Everything is orders of magnitude bigger in terms of hyperparameters, dozens of optimizations are made across many
areas, but it remains the same architecture.

Additionally, it's a massive infrastructure and compute challenge to train models of that size

Second, after pre-training: OpenAI has a big stupid document autocompleter, it would blabber in many languages and
complete things for you; but it isn't yet useful. It may answer your question, it may complete it with another question,
it may rephrase it, it may start writing a blogpost, it may answer with a no, etc.

This is where finetuning and "alignment" comes into play (Refer OpenAI's Optimizing Language Models for Dialogue) where
we basically turn our language model into an assistant.

This is where the finetuning step comes into play. We finetune the model by training it on documents that look like what
an assistant would behave like. This is on the order of thousands of documents and this is often very effective at
steering the model towards "assistant-like behavior", there are other steps such as adding a reward model, and
reinforcement
learning steps that happen later.

The data used for finetuning is proprietary to OpenAI and overall the finetuning steps are much harder to replicate than
the initial pretraining steps.

Trained 10.8m parameter model on RTX 3050 with:

```python
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200
n_embed = 384
n_head = 6
n_layer = 6
dropout = 0.2
```

Trained 25m parameter model on A100 with:

```python
batch_size = 128
block_size = 512
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200
n_embed = 512
n_head = 8
n_layer = 8
dropout = 0.2
```