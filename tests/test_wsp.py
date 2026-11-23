"""Unit tests for weight-space programming helpers."""
import numpy as np
from sklearn.neural_network import MLPRegressor

from src.train_wsp import (
    average_weights,
    clone_weights,
    dare_lite,
    mix_weights,
    multi_task_arith,
    set_weights,
    ties_lite,
    ties_merge,
)


def _tiny_mlp():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 4))
    y = X[:, 0] * 0.5 + rng.normal(scale=0.1, size=40)
    mlp = MLPRegressor(hidden_layer_sizes=(8,), max_iter=200, random_state=0)
    mlp.fit(X, y)
    return mlp


def test_mix_weights_at_zero_keeps_base():
    mlp = _tiny_mlp()
    base = clone_weights(mlp)
    other = (
        [c + 1.0 for c in base[0]],
        [b + 1.0 for b in base[1]],
    )
    mixed = mix_weights(base, other, alpha=0.0)
    for a, b in zip(mixed[0], base[0]):
        assert np.allclose(a, b)
    for a, b in zip(mixed[1], base[1]):
        assert np.allclose(a, b)


def test_mix_weights_at_one_reaches_other():
    mlp = _tiny_mlp()
    base = clone_weights(mlp)
    other = (
        [c + 2.0 for c in base[0]],
        [b + 2.0 for b in base[1]],
    )
    mixed = mix_weights(base, other, alpha=1.0)
    for a, b in zip(mixed[0], other[0]):
        assert np.allclose(a, b)


def test_average_weights_is_midpoint():
    mlp = _tiny_mlp()
    a = clone_weights(mlp)
    b = ([c + 2.0 for c in a[0]], [i + 2.0 for i in a[1]])
    mid = average_weights([a, b])
    for m, left in zip(mid[0], a[0]):
        assert np.allclose(m, left + 1.0)


def test_set_weights_round_trip():
    mlp = _tiny_mlp()
    weights = clone_weights(mlp)
    weights[0][0] += 0.5
    set_weights(mlp, *weights)
    again = clone_weights(mlp)
    assert np.allclose(again[0][0], weights[0][0])


def test_ties_lite_zeros_tiny_deltas():
    mlp = _tiny_mlp()
    base = clone_weights(mlp)
    other = (
        [c.copy() for c in base[0]],
        [b.copy() for b in base[1]],
    )
    flat_size = sum(c.size for c in other[0]) + sum(b.size for b in other[1])
    other[0][0].ravel()[0] += 10.0
    other[0][0].ravel()[1] += 0.5
    # Keep only the single largest-magnitude delta.
    trimmed = ties_lite(base, other, keep=1.0 / flat_size, alpha=1.0)
    assert abs(trimmed[0][0].ravel()[0] - (base[0][0].ravel()[0] + 10.0)) < 1e-6
    assert abs(trimmed[0][0].ravel()[1] - base[0][0].ravel()[1]) < 1e-6


def test_multi_task_arith_adds_both_deltas():
    mlp = _tiny_mlp()
    base = clone_weights(mlp)
    t1 = ([c + 2.0 for c in base[0]], [b + 2.0 for b in base[1]])
    t2 = ([c + 4.0 for c in base[0]], [b + 4.0 for b in base[1]])
    # 0.5*Δ1 + 0.25*Δ2 = 0.5*2 + 0.25*4 = 2
    out = multi_task_arith(base, [t1, t2], [0.5, 0.25])
    for m, left in zip(out[0], base[0]):
        assert np.allclose(m, left + 2.0)


def test_dare_lite_is_deterministic():
    mlp = _tiny_mlp()
    base = clone_weights(mlp)
    other = ([c + 1.0 for c in base[0]], [b + 1.0 for b in base[1]])
    a = dare_lite(base, other, drop=0.5, alpha=1.0, seed=3)
    b = dare_lite(base, other, drop=0.5, alpha=1.0, seed=3)
    for x, y in zip(a[0], b[0]):
        assert np.allclose(x, y)


def test_ties_merge_keeps_agreeing_sign():
    mlp = _tiny_mlp()
    base = clone_weights(mlp)
    a = ([c.copy() for c in base[0]], [b.copy() for b in base[1]])
    b = ([c.copy() for c in base[0]], [i.copy() for i in base[1]])
    a[0][0].ravel()[0] += 4.0
    b[0][0].ravel()[0] += 2.0
    merged = ties_merge(base, [a, b], keep=1.0, alpha=1.0)
    # Both positive → elect +, average magnitude 3
    assert abs(merged[0][0].ravel()[0] - (base[0][0].ravel()[0] + 3.0)) < 1e-6
