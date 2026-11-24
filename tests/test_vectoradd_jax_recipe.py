# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Traced-vs-callback parity for the ``vectoradd_jax`` example.

The example mixes a static input (``norm_ord``) with nested submodels, which
no other traced test Tesseract does.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def _assert_trees_close(a, b) -> None:
    jax.tree.map(lambda x, y: np.testing.assert_allclose(x, y, rtol=1e-5), a, b)


@pytest.mark.parametrize("use_jit", [True, False])
def test_apply_matches_between_dispatch_modes(
    vectoradd_apply, vectoradd_jax_ab, use_jit
):
    def f(traceable):
        fn = lambda ab: vectoradd_apply(ab, traceable)
        if use_jit:
            fn = jax.jit(fn)
        return fn(vectoradd_jax_ab)

    _assert_trees_close(f(True), f(False))


@pytest.mark.parametrize("use_jit", [True, False])
def test_jvp_matches_between_dispatch_modes(vectoradd_apply, vectoradd_jax_ab, use_jit):
    tangents = jax.tree.map(jnp.ones_like, vectoradd_jax_ab)

    def f(traceable):
        full = lambda ab: vectoradd_apply(ab, traceable)
        if use_jit:
            full = jax.jit(full)
        return jax.jvp(full, (vectoradd_jax_ab,), (tangents,))[1]

    _assert_trees_close(f(True), f(False))


@pytest.mark.parametrize("use_jit", [True, False])
def test_grad_matches_between_dispatch_modes(
    vectoradd_apply, vectoradd_jax_ab, use_jit
):
    def loss(ab, traceable):
        out = vectoradd_apply(ab, traceable)
        return (
            out["vector_add"]["result"].sum() + out["vector_min"]["normed_result"].sum()
        )

    def f(traceable):
        fn = lambda ab: loss(ab, traceable)
        if use_jit:
            fn = jax.jit(fn)
        return jax.grad(fn)(vectoradd_jax_ab)

    _assert_trees_close(f(True), f(False))


@pytest.mark.parametrize("jac_direction", ["fwd", "rev"])
def test_jacobian_matches_between_dispatch_modes(
    vectoradd_apply, vectoradd_jax_ab, jac_direction
):
    jac_fn = jax.jacfwd if jac_direction == "fwd" else jax.jacrev

    def out_fn(ab, traceable):
        return vectoradd_apply(ab, traceable)["vector_add"]["result"]

    direct = jac_fn(lambda ab: out_fn(ab, True))(vectoradd_jax_ab)
    callback = jac_fn(lambda ab: out_fn(ab, False))(vectoradd_jax_ab)
    _assert_trees_close(direct, callback)


def test_static_norm_ord_affects_result_under_traceable(
    vectoradd_apply, vectoradd_jax_ab
):
    """``norm_ord`` stays a static input next to array-carrying submodels."""
    out_ord2 = vectoradd_apply(vectoradd_jax_ab, traceable=True, norm_ord=2)
    out_ord1 = vectoradd_apply(vectoradd_jax_ab, traceable=True, norm_ord=1)
    assert not np.allclose(
        out_ord2["vector_add"]["normed_result"],
        out_ord1["vector_add"]["normed_result"],
    )
