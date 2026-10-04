"""Tests for attention.py.

Run with:  python test_attention.py       (no dependencies)
       or:  python -m pytest

These check the exact mathematical properties attention must satisfy — the kind
of bug that would otherwise only show up as a model that mysteriously won't learn.
"""

import math

from attention import (
    transpose, matmul, softmax, causal_mask,
    scaled_dot_product_attention, SelfAttention, MultiHeadAttention,
)


def approx(a, b, tol=1e-9):
    return abs(a - b) < tol


# ---- linear algebra ----
def test_transpose():
    assert transpose([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]]


def test_matmul():
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    assert matmul(A, B) == [[19, 22], [43, 50]]


# ---- softmax ----
def test_softmax_sums_to_one_and_positive():
    out = softmax([1.0, 2.0, 3.0])
    assert approx(sum(out), 1.0)
    assert all(x > 0 for x in out)
    # Bigger input -> bigger probability.
    assert out[2] > out[1] > out[0]


def test_softmax_is_numerically_stable():
    # Without the max-subtraction trick, exp(1000) overflows to infinity.
    out = softmax([1000.0, 1000.0, 1000.0])
    assert all(approx(x, 1 / 3) for x in out)


def test_softmax_handles_negative_infinity():
    # A masked (-inf) position must get exactly zero probability.
    out = softmax([1.0, float("-inf"), 1.0])
    assert approx(out[1], 0.0)
    assert approx(out[0] + out[2], 1.0)


# ---- causal mask ----
def test_causal_mask_is_lower_triangular():
    m = causal_mask(3)
    assert m == [
        [True, False, False],
        [True, True, False],
        [True, True, True],
    ]


# ---- attention core ----
def test_attention_weights_are_distributions():
    Q = [[1.0, 0.0], [0.0, 1.0]]
    K = [[1.0, 0.0], [0.0, 1.0]]
    V = [[1.0, 2.0], [3.0, 4.0]]
    _, weights = scaled_dot_product_attention(Q, K, V)
    for row in weights:
        assert approx(sum(row), 1.0)        # each row sums to 1
        assert all(w >= 0 for w in weights[0])


def test_attention_output_shape():
    Q = [[1.0, 0.0, 0.0]] * 4   # 4 tokens, d_k=3
    K = [[1.0, 0.0, 0.0]] * 4
    V = [[1.0, 2.0]] * 4        # d_v=2
    out, weights = scaled_dot_product_attention(Q, K, V)
    assert len(out) == 4 and len(out[0]) == 2       # (n x d_v)
    assert len(weights) == 4 and len(weights[0]) == 4  # (n x n)


def test_causal_mask_blocks_the_future():
    Q = [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
    K = [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
    V = [[1.0], [2.0], [3.0]]
    _, weights = scaled_dot_product_attention(Q, K, V, mask=causal_mask(3))
    # Upper triangle (future positions) must be exactly zero.
    assert approx(weights[0][1], 0.0) and approx(weights[0][2], 0.0)
    assert approx(weights[1][2], 0.0)
    # First token can only attend to itself -> weight 1.0 there.
    assert approx(weights[0][0], 1.0)


def test_one_hot_attention_returns_that_value():
    """If a query matches one key far more strongly, the output is that Value."""
    # Query identical to key 1, and keys are very distinct, so attention on
    # position 1 dominates. Large magnitudes make the softmax nearly one-hot.
    Q = [[0.0, 10.0]]
    K = [[10.0, 0.0], [0.0, 10.0]]   # key 1 aligns with the query
    V = [[1.0, 1.0], [9.0, 9.0]]
    out, weights = scaled_dot_product_attention(Q, K, V)
    assert weights[0][1] > 0.99, "query should lock onto the matching key"
    assert approx(out[0][0], 9.0, tol=0.1) and approx(out[0][1], 9.0, tol=0.1)


def test_query_attends_most_to_most_similar_key():
    # Three orthogonal keys; the query points along key 2, so it should win.
    Q = [[0.0, 0.0, 5.0]]
    K = [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]]
    V = [[1.0], [2.0], [3.0]]
    _, weights = scaled_dot_product_attention(Q, K, V)
    best = max(range(3), key=lambda j: weights[0][j])
    assert best == 2, "query should attend most to the aligned key"


# ---- layers ----
def test_self_attention_layer_shapes():
    layer = SelfAttention(d_model=8, seed=3)
    X = [[0.1 * i] * 8 for i in range(5)]   # 5 tokens, d_model=8
    out, weights = layer.forward(X, causal=True)
    assert len(out) == 5 and len(out[0]) == 8
    assert len(weights) == 5 and len(weights[0]) == 5


def test_multihead_concatenates_heads():
    mha = MultiHeadAttention(d_model=8, num_heads=4, seed=1)
    X = [[0.1 * i] * 8 for i in range(3)]
    out, all_weights = mha.forward(X)
    # 4 heads of size 2 each -> concatenated back to d_model=8.
    assert len(out) == 3 and len(out[0]) == 8
    assert len(all_weights) == 4   # one weight matrix per head


def test_multihead_requires_divisible_dims():
    try:
        MultiHeadAttention(d_model=10, num_heads=3)
        raise AssertionError("should reject heads that don't divide d_model")
    except AssertionError as e:
        assert "divide" in str(e)


if __name__ == "__main__":
    import traceback

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for test in tests:
        try:
            test()
            print(f"PASS  {test.__name__}")
            passed += 1
        except Exception:
            print(f"FAIL  {test.__name__}")
            traceback.print_exc()
    print(f"\n{passed}/{len(tests)} tests passed")
