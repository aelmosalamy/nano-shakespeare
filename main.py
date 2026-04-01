import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

TRAIN = False

if TRAIN:
    torch.manual_seed(1337)

# hyperparameters
batch_size = 64  # how many independent sequences will be process in parallel
block_size = 256  # the maximum context length for predictions
max_iters = 5000 if TRAIN else 0
eval_interval = 500
# very low learning_rate because self-attention cant tolerate high learning rates, we compensate for that by increasing
# the number of iterations
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embed = 384
n_head = 6
n_layer = 6
dropout = 0.2


if device == "cuda":
    print(f"[+] gpu detected: {torch.cuda.get_device_name(0)}!")
else:
    print("[-] falling back to cpu.")

DATASET_PATH = "dataset/tinyshakespeare.txt"

# load our dataset
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    text = f.read()


# our vocabulary is determined by the dataset, these are the "tokens" our model is able to emit
# since we are building a character-level language model, this means our self-defined "token" is
# a single character. hence we will map every character to an integer
vocab = sorted(list(set(text)))
vocab_size = len(vocab)

# map characters to integers
stoi = {c: i for i, c in enumerate(vocab)}
itos = {i: c for i, c in enumerate(vocab)}
# helper encode function
encode = lambda s: [stoi[c] for c in s]
# helper decode function
decode = lambda l: "".join(itos[i] for i in l)

# note that different, more sophisticated encoding schemes exist. they all address certain production problems but
# at the end of the day, all they do is convert text to integers. we call them tokenizers. they do not map char->int
# as we are doing here, but they do not encode complete words either, they encode sub-word units.
# examples tokenizers include Google's SentencePiece, OpenAI's tiktoken

# let's encode our dataset and wrap it inside a tensor
data = torch.tensor(encode(text), dtype=torch.long).to(device)

# before moving forward, let's split our dataset into training and validation
# we will train on 90% and use the remaining 10% to validate that our model is working correctly.
# this will help us understand to what extent is our model *overfitting*, we do not want perfect memorization of the
# exact shakespeare dataset; instead, we want it to be able to generate shakespeare-like data!
split_at = int(0.9 * len(text))
train_data = data[:split_at]
validation_data = data[split_at:]


# let's now plugin what we have into the transformer so it can learn the patterns within our dataset
# notice that we will never feed the entire dataset into the transformer, that would be computationally infeasible;
# instead, we will chunk our data and feed the transformer one chunk at a time.

# note that we do not chunk just for efficiency, but also for learning reasons. we want the model to learn the
# probabilities of anything between 1 and up to the context length.

# when we train on a block of size 8, the transformer will simultaneously learn about all the probabilities within,
# whether it is from 1 to 2 or from 2 to 3 or so 7 to 8.

# we need to keep the gpu busy during training, so other than the block_size, we will have a batch_size which determines
# how many batches of independent sequences do we train simultaneously; emphasis on independent. they do not talk to
# each other.


def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else validation_data
    # we generate batch_size random offsets within our data
    ix = torch.randint(len(data) - block_size, size=(batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


# this function averages the loss across multiple batches
# the no_grad pytorch manager tells pytorch that within this function, we won't be calling backward on.
# this makes pytorch must more memory efficient as it won't be storing the intermediate steps since we will never do
# backpropagation here.
@torch.no_grad()
def estimate_loss():
    out = {}
    # reset model to evaluation phase
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    # set model back to train phase, currently we don't have any difference since our model does not implement any
    model.train()
    return out


class FeedForward(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            # the 4-factor improves loss
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            # adding the projection layer going back through the residual pathway (don't understand waht that means)
            nn.Linear(4 * n_embed, n_embed),
            # dropout takes a few nodes in a NN and randomly shuts them (drops them 0) every forward/backwards pass
            # this effectively trains on a sample of subnetworks
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # so far the tokens are purely consecutive, there is no coupling between the tokens.
        # e.g. how to make the 5th token communicate with the 4th, 3rd, 2nd, 1st tokens? a very naive simple way would be to average
        # the previous tokens to give me a future vector of what the 5th token is within its context (as far as i understood)

        # important: attention is a communication mechanism

        # math trick to do self-attention via matrix multiplication which makes it way faster

        # if i am a vowel, perhaps i want to know what the consonant that came before me, i need to know the data in the past,
        # this is the problem which self-attention solves. every single node/token will emit two vectors: a query and a key.
        # the query vector is what i am looking for
        # the key vector is: what do i contain
        # if we do a dot product between the keys and the queries which gives us the weights, if a key and a query are aligned,
        # they will interact with each other allowing a key to learn more/have higher affinity with a specific previous token
        # than any other token in that sequence

        B, T, C = x.shape

        # we will implement a single head of self-attention
        k = self.key(x)  # (B, T, 16)
        q = self.query(x)  # (B, T, 16)
        # no communication between key and query has been done till now. we just forwarded the key and query on x.
        # compute attention scores "affinities", we normalize by dividing by sqrt(C) AKA sqrt(head_size)?
        # this scaled attention is an important normalization step as seen in "Attention is All You Need".
        wei = (
            q @ k.transpose(-2, -1) * C**-0.5
        )  # (B, T, 16) @ (B, 16, T) ---> (B, T, T)

        # we mask via our triangular matrix to make sure it does not communicate with the past (this makes it a decoder block)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        # v is the values that we aggregate rather than the aggregate x, x is private information
        # additionally, these nodes are doing self-attention because key(x) and value(x) are based on the same node; attention
        # is more general than that. cross-attention is when there are a bunch of other nodes that we would want to interact with.
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        # parallel self-attention heads, allowing us to have num_heads communication channels in parallel,
        # multi-headed self-attention is similar to convolution groups. since our tokens have lots to talk about, it
        # helps to create multiple independent channels of communication, gather different kinds of interactions,
        # then combine the output.
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)

    def forward(self, x):
        # concatenate the output of all heads in parallel over the channel dimension (dim=-1)
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out


# we use this transformer block to capture the communication step (done via self-attention mechanism) as well as the
# computation step done by the feed-forward linear + non-linear layers.
# we are implementing only the decoder part, without the encoder or cross-attention components
class Block(nn.Module):
    """transformer block: communication followed by communication"""

    def __init__(self, n_embed, n_head):
        # n_embed: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        block_head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, block_head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        # x + ... is an expression that adds residual optimization
        # we do pre normalization by adding the layer norm

        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# let's start feeding our inputs to our transformer neural network, we will feed this to the simplest language model:
# Bigram Language Model. we can refer to andrej karpathy's Make More series where this model is covered more in-depth.
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        # we plug a self-attention component into our network via our Head class
        # single self-attention Head
        # self.sa_head = Head(n_embed)
        self.blocks = nn.Sequential(
            *[Block(n_embed, n_head=n_head) for _ in range(n_layer)],
        )  # 3 blocks, each with 4 heads of 8-dimensional self-attention (32 / 4 = 8)
        self.ln_f = nn.LayerNorm(n_embed)  # final layer norm
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B, T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B: batch, T: time, C: channel)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=device)
        )  # (T, C)
        # x holds not only the token identities but also the position embeddings.
        # this information is currently not useful due to our bigram model
        x = tok_emb + pos_emb  # (B, T, C)
        x = self.blocks(x)  # (B, T, C)
        x = self.ln_f(x)  # (B, T, C)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            # reshape our logits and targets to match what pytorch expects
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            # negative log likelihood loss (implemented as cross_entropy in pytorch
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # must make sure that our idx which is fed into the model never exceeds block_size to prevent our positional
            # embeddings from running out of scope
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


def generate_text(prompt, max_new_tokens=200):
    model.eval()

    context = torch.tensor([encode(prompt)], dtype=torch.long).to(device)

    with torch.no_grad():
        output = model.generate(context, max_new_tokens=max_new_tokens)

    return decode(output[0].tolist())


if __name__ == "__main__":
    MODEL_VARIATION = "10m"

    if len(sys.argv) == 2:
        MODEL_VARIATION = sys.argv[1]
    assert MODEL_VARIATION in ["10m", "25m", "25m_overfit"], (
        "You must choose between the `10m` and `25m` parameter models"
    )

    if MODEL_VARIATION.startswith("25m"):
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

    CHECKPOINT_PATH = f"checkpoints/checkpoint_{MODEL_VARIATION}.pth"

    model = BigramLanguageModel()
    model.to(device)

    # use AdamW optimizer, a nice advanced optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    if os.path.exists(CHECKPOINT_PATH):
        print("[+] loading checkpoint...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

        # override hyperparameters from checkpoint if they were saved there
        if "hyperparameters" in checkpoint:
            hp = checkpoint["hyperparameters"]
            vocab_size = hp["vocab_size"]
            n_embed = hp["n_embed"]
            n_head = hp["n_head"]
            n_layer = hp["n_layer"]
            block_size = hp["block_size"]
            dropout = hp["dropout"]

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        start_iter = checkpoint["iter"]
        print(f"[+] loaded at iteration {start_iter}")
    else:
        print("[!] no checkpoint found, starting fresh")
        start_iter = 0

    print(
        f"[+] model has {sum(p.numel() for p in model.parameters()) / 1e6:.03f}m parameters"
    )
    loss = None
    for itr in range(start_iter, max_iters + 1):
        if not TRAIN:
            print("[!] skipping training")
            break

        # every once in a while evaluate the loss on train and val sets
        if itr % eval_interval == 0:
            losses = estimate_loss()
            print(
                f"step {itr:04}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "iter": itr,
                    "hyperparameters": {
                        "vocab_size": vocab_size,
                        "n_embed": n_embed,
                        "n_head": n_head,
                        "n_layer": n_layer,
                        "block_size": block_size,
                        "dropout": dropout,
                    },
                },
                CHECKPOINT_PATH,
            )
            print(f"[+] checkpoint saved at iter {itr}")

        # sample a batch of data
        xb, yb = get_batch("train")

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    if loss:
        print(f"[-] loss after optimization: {loss.item():.4f}")
    print("[*] sample generation:")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
