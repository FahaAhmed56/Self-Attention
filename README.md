# Self-Attention From Scratch

The mechanism at the heart of every transformer — **self-attention** — built in
pure Python with **no libraries, not even NumPy**. This is the "Attention" of
*Attention Is All You Need* (2017), and it's what lets GPT, Llama, and Claude
understand how words relate to each other.

Pure standard library — **nothing to install**.

## What attention does

When a model reads "the cat sat because **it** was tired," something has to tell
it that "it" refers to "cat." That something is attention: each word looks at
the other words and pulls in the ones that matter for understanding it.

The recipe, called **scaled dot-product attention**, gives every token three
vectors:

- **Query (Q)** — "what am I looking for?"
- **Key (K)** — "what do I offer?"
- **Value (V)** — "what I'll contribute if you attend to me"

Then, for each token:

1. Compare its **Query** to every token's **Key** with a dot product (similarity).
2. Scale by `1/√d_k` so large vectors don't blow up the numbers.
3. **Softmax** those scores into weights that sum to 1 (a probability distribution).
4. Output a **weighted sum of all the Values** using those weights.

So each token's new representation is a blend of the tokens it cared about. A
**causal mask** stops a token from attending to later tokens — which is exactly
what lets a model generate text one word at a time.

## See it work

Needs [Python](https://www.python.org) 3.8+.

```bash
python attention.py demo
```

Real output — the sentence "the cat sat on the mat" with a causal mask:

```
        the   cat   sat    on   the   mat
 the |  1.00  0.00  0.00  0.00  0.00  0.00
 cat |  0.38  0.62  0.00  0.00  0.00  0.00
 sat |  0.27  0.27  0.45  0.00  0.00  0.00
  on |  0.23  0.23  0.27  0.26  0.00  0.00
 the |  0.26  0.16  0.16  0.16  0.26  0.00
 mat |  0.14  0.16  0.14  0.16  0.14  0.24
```

Two things to notice:

- **The second "the" (row 5) attends to the first "the"** — 0.26 on position 0,
  more than it gives any other earlier word. Identical words have identical
  vectors, so their dot product is high, so attention links them. That's context
  being captured.
- **The zeros in the upper-right triangle** are the causal mask: no token attends
  to a token that comes after it.

## Use it in code

```python
from attention import scaled_dot_product_attention, causal_mask, MultiHeadAttention

out, weights = scaled_dot_product_attention(Q, K, V, mask=causal_mask(len(Q)))

# Or a full multi-head layer with learned projections:
mha = MultiHeadAttention(d_model=64, num_heads=8)
out, per_head_weights = mha.forward(X, causal=True)
```

## Why multi-head?

Real transformers run several attention "heads" in parallel and concatenate
them. Each head can specialise — one might track subject→verb links, another
might focus on the previous word or on long-range references. `MultiHeadAttention`
does this: it splits `d_model` into `num_heads` smaller heads, attends in each,
and joins the results.

## Run the tests

```bash
python test_attention.py       # no dependencies
# or, with pytest:
python -m pytest
```

They verify the maths that has to be exactly right: softmax is a valid
distribution (sums to 1, numerically stable, treats `-inf` masks as zero), the
causal mask is lower-triangular and truly blocks the future, attention weights
are distributions, a one-hot query returns exactly that token's Value, a query
attends most to its most-similar key, and the multi-head layer concatenates to
the right shape.

## How it's built

- `transpose`, `matmul`, `softmax` — the minimal linear algebra, by hand
- `causal_mask` — the lower-triangular "no peeking ahead" mask
- `scaled_dot_product_attention` — the four-step recipe above
- `SelfAttention` — one head: learned Q/K/V projections, then attention
- `MultiHeadAttention` — several heads in parallel, concatenated

## Where this sits in an LLM

```
text → [tokenizer] → tokens → [embeddings] → [ATTENTION × N layers] → next-token prediction
```

This project is the attention block. Companion from-scratch projects cover the
**tokenizer** (text → tokens) and a **neural network with backprop** (how the
weights are learned). Together they take the mystery out of what an LLM actually
does under the API call.

## Ideas for later

- Add the rest of a transformer block: a feed-forward layer, residual
  connections, and layer normalisation
- Stack blocks and train a tiny character-level language model end to end
- Add **positional encodings** so the model knows word order (attention alone is
  order-blind)
- Switch the matrix ops to NumPy and compare speed
