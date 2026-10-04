#!/usr/bin/env python3
"""Self-attention from scratch — the heart of the transformer. No libraries.

Attention is the mechanism that lets a language model understand context. When
it processes the word "it" in "the cat sat because IT was tired", attention is
what lets "it" look back at "cat" and borrow its meaning. Every modern LLM —
GPT, Llama, Claude — is a stack of attention layers. This file builds one by
hand so the mechanism is no longer a mystery.

THE RECIPE (scaled dot-product attention). Each token produces three vectors:
    Query (Q)  - "what am I looking for?"
    Key   (K)  - "what do I offer?"
    Value (V)  - "what will I pass along if attended to?"

For each token we:
    1. compare its Query to every token's Key with a dot product  (similarity)
    2. scale by 1/sqrt(d_k) so the numbers don't explode
    3. softmax those scores into weights that sum to 1  (a distribution)
    4. take a weighted sum of all the Values using those weights

The output for each token is a blend of the Values of the tokens it cared about.
A CAUSAL mask stops a token from attending to tokens that come after it, which
is what lets a model generate text left to right.

Everything is plain Python lists (matrices as lists of lists), so the maths is
fully visible — no NumPy hiding the work.

CLI:
    python attention.py demo    # watch a repeated word attend to its earlier twin
"""

import argparse
import math
import random


# ---------------------------------------------------------------------------
# Minimal linear algebra (matrices = list of lists of floats).
# ---------------------------------------------------------------------------
def transpose(M):
    return [list(row) for row in zip(*M)]


def matmul(A, B):
    """Standard matrix product: (n x k) @ (k x m) -> (n x m)."""
    assert len(A[0]) == len(B), "inner dimensions must match"
    Bt = transpose(B)  # so each column of B becomes a row we can dot with
    return [[sum(a * b for a, b in zip(row, col)) for col in Bt] for row in A]


def softmax(row):
    """Turn a list of scores into a probability distribution (sums to 1).

    We subtract the max first for numerical stability — exp of a big number
    overflows, and subtracting the max changes nothing mathematically.
    """
    m = max(row)
    exps = [math.exp(x - m) for x in row]  # exp(-inf) == 0, so masks vanish here
    total = sum(exps)
    return [e / total for e in exps]


# ---------------------------------------------------------------------------
# The core: scaled dot-product attention.
# ---------------------------------------------------------------------------
def causal_mask(n):
    """mask[i][j] is True if token i is ALLOWED to attend to token j.

    Lower-triangular: a token may look at itself and earlier tokens, never later.
    """
    return [[j <= i for j in range(n)] for i in range(n)]


def scaled_dot_product_attention(Q, K, V, mask=None):
    """The whole mechanism. Returns (output, attention_weights).

    Q: (n x d_k)   K: (n x d_k)   V: (n x d_v)
    output: (n x d_v)   attention_weights: (n x n)
    """
    d_k = len(Q[0])
    scale = 1.0 / math.sqrt(d_k)

    # 1 & 2: similarity of every query to every key, scaled.
    scores = matmul(Q, transpose(K))
    scores = [[s * scale for s in row] for row in scores]

    # Optional mask: block disallowed positions by sending them to -inf, so
    # softmax gives them exactly zero weight.
    if mask is not None:
        scores = [
            [score if mask[i][j] else float("-inf") for j, score in enumerate(row)]
            for i, row in enumerate(scores)
        ]

    # 3: softmax each row into weights that sum to 1.
    weights = [softmax(row) for row in scores]

    # 4: each token's output is the weighted sum of all Values.
    output = matmul(weights, V)
    return output, weights


# ---------------------------------------------------------------------------
# A learnable self-attention layer: project the input into Q, K, V.
# ---------------------------------------------------------------------------
def _random_matrix(rows, cols, rng, scale=0.5):
    return [[rng.uniform(-scale, scale) for _ in range(cols)] for _ in range(rows)]


class SelfAttention:
    """One attention head. X -> (Q,K,V via learned weights) -> attention."""

    def __init__(self, d_model, d_k=None, seed=0):
        self.d_model = d_model
        self.d_k = d_k or d_model
        rng = random.Random(seed)
        # Each projection turns a d_model vector into a d_k vector.
        self.Wq = _random_matrix(d_model, self.d_k, rng)
        self.Wk = _random_matrix(d_model, self.d_k, rng)
        self.Wv = _random_matrix(d_model, self.d_k, rng)

    def forward(self, X, causal=False):
        """X is (seq_len x d_model). Returns (output, attention_weights)."""
        Q = matmul(X, self.Wq)
        K = matmul(X, self.Wk)
        V = matmul(X, self.Wv)
        mask = causal_mask(len(X)) if causal else None
        return scaled_dot_product_attention(Q, K, V, mask)


class MultiHeadAttention:
    """Several attention heads in parallel, their outputs concatenated.

    Different heads can learn to look for different kinds of relationships
    (one for subjects/verbs, another for nearby words, etc.) — which is why
    real transformers use many heads rather than one big one.
    """

    def __init__(self, d_model, num_heads, seed=0):
        assert d_model % num_heads == 0, "d_model must divide evenly into heads"
        self.num_heads = num_heads
        d_k = d_model // num_heads
        self.heads = [SelfAttention(d_model, d_k, seed=seed + h) for h in range(num_heads)]

    def forward(self, X, causal=False):
        outputs = []
        all_weights = []
        for head in self.heads:
            out, w = head.forward(X, causal=causal)
            outputs.append(out)
            all_weights.append(w)
        # Concatenate the heads' outputs along the feature dimension.
        combined = [sum((outputs[h][i] for h in range(self.num_heads)), [])
                    for i in range(len(X))]
        return combined, all_weights


# ---------------------------------------------------------------------------
# Demo: show attention letting a repeated word look back at its earlier self.
# ---------------------------------------------------------------------------
def _print_attention(tokens, weights):
    width = max(len(t) for t in tokens) + 1
    header = " " * (width + 2) + " ".join(f"{t:>5}" for t in tokens)
    print(header)
    for i, row in enumerate(weights):
        cells = " ".join(f"{w:5.2f}" for w in row)
        print(f"{tokens[i]:>{width}} | {cells}")


def cmd_demo(args):
    # A sentence with a repeated word. We hand-build embeddings so that
    # identical words get identical vectors — making attention easy to read.
    tokens = ["the", "cat", "sat", "on", "the", "mat"]
    vocab_vectors = {
        "the": [1.0, 0.0, 0.0, 0.0],
        "cat": [0.0, 1.0, 0.0, 0.0],
        "sat": [0.0, 0.0, 1.0, 0.0],
        "on":  [0.0, 0.0, 0.3, 0.3],
        "mat": [0.0, 0.3, 0.0, 1.0],
    }
    X = [vocab_vectors[t] for t in tokens]

    print("Sentence:", " ".join(tokens))
    print("\nWe use the embeddings directly as Q, K, V (identity projection) so the")
    print("attention pattern is easy to interpret.\n")

    # Identity projection: Q=K=V=X, with a causal mask (generation-style).
    _, weights = scaled_dot_product_attention(X, X, X, mask=causal_mask(len(X)))

    print("Causal self-attention weights (row = token doing the looking,")
    print("column = token being looked at; each row sums to 1):\n")
    _print_attention(tokens, weights)

    # Point out the headline: the 2nd 'the' (index 4) attends to the 1st (index 0).
    second_the = 4
    w = weights[second_the]
    looked_at = max(range(len(tokens)), key=lambda j: w[j])
    print(f"\nThe second 'the' (position {second_the}) attends most to position "
          f"{looked_at} ('{tokens[looked_at]}'), weight {w[looked_at]:.2f}.")
    print("Zeros in the upper-right triangle show the causal mask working:")
    print("no token can attend to a token that comes after it.\n")

    # And show multi-head on random projections just to prove the plumbing.
    mha = MultiHeadAttention(d_model=4, num_heads=2, seed=1)
    out, _ = mha.forward(X, causal=True)
    print(f"Multi-head (2 heads) output shape: {len(out)} tokens x {len(out[0])} features.")


def main():
    parser = argparse.ArgumentParser(description="Self-attention from scratch.")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("demo", help="visualise attention on a short sentence")
    p.set_defaults(func=cmd_demo)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
