# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parity tests for GPU-direct Tesseract dispatch (Strategies A and B).

These require a real GPU and a served (subprocess) GPU Tesseract, since CUDA IPC
is cross-process and cannot be self-opened. They are marked ``gpu`` and skipped
where CUDA / CuPy / a GPU-backed JAX are unavailable.

Both strategies are checked against:
* an analytic reference (``c = a*scale + b``), and
* the existing host-callback baseline (``apply_tesseract``), which is the
  behavior the GPU-direct path must reproduce bit-for-bit.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("cupy")

from tesseract_core import Tesseract  # noqa: E402

from tests_gpu.serve_util import serve_gpu_tesseract  # noqa: E402

pytestmark = pytest.mark.gpu


def _gpu_available() -> bool:
    try:
        return any(d.platform == "gpu" for d in jax.devices())
    except Exception:
        return False


if not _gpu_available():
    pytest.skip("no GPU backend for JAX", allow_module_level=True)


def _to_np(x):
    get = getattr(x, "get", None)
    return get() if callable(get) else np.asarray(x)


@pytest.fixture(scope="module")
def gpu_tesseract():
    # Serve with the default base64 output; the GPU-direct strategies opt into
    # cuda_ipc per call, so this also exercises that opt-in path end-to-end.
    with serve_gpu_tesseract(output_format="json+base64") as url:
        yield Tesseract.from_url(url)


@pytest.mark.parametrize("impl", ["A", "B"])
@pytest.mark.parametrize("n", [1, 8, 1000, 100_003])
def test_apply_matches_analytic(gpu_tesseract, impl, n):
    from tesseract_jax_gpu.frontend import apply_gpu_direct

    a = jnp.arange(n, dtype=jnp.float32)
    b = jnp.ones(n, dtype=jnp.float32) * 3.0
    out = apply_gpu_direct(gpu_tesseract, {"a": a, "b": b}, impl=impl)
    expected = np.asarray(a) * 2.0 + np.asarray(b)
    np.testing.assert_allclose(_to_np(out["c"]), expected, rtol=1e-6, atol=0)


@pytest.mark.parametrize("impl", ["A", "B"])
def test_apply_matches_host_callback_baseline(gpu_tesseract, impl):
    """GPU-direct must reproduce the existing host-callback path exactly."""
    from tesseract_jax import apply_tesseract

    from tesseract_jax_gpu.frontend import apply_gpu_direct

    a = jnp.linspace(-5, 5, 257, dtype=jnp.float32)
    b = jnp.linspace(10, -10, 257, dtype=jnp.float32)

    # Baseline uses a *fresh* client so its default (base64) transport is not
    # affected by any GPU-direct state on the shared fixture client.
    baseline_client = Tesseract.from_url(gpu_tesseract._client.url)
    baseline = apply_tesseract(baseline_client, {"a": a, "b": b})
    direct = apply_gpu_direct(gpu_tesseract, {"a": a, "b": b}, impl=impl)

    np.testing.assert_array_equal(_to_np(direct["c"]), _to_np(baseline["c"]))


@pytest.mark.parametrize("impl", ["A", "B"])
def test_serial_reuse_ring1(gpu_tesseract, impl):
    """Back-to-back serial dispatches: exercises the ring-1 lifetime contract.

    Each output must be copied out before the next request releases it; a
    correctness bug here would surface as stale/overwritten data on later calls.
    """
    from tesseract_jax_gpu.frontend import apply_gpu_direct

    for i in range(20):
        a = jnp.full((512,), float(i), dtype=jnp.float32)
        b = jnp.full((512,), float(2 * i), dtype=jnp.float32)
        out = apply_gpu_direct(gpu_tesseract, {"a": a, "b": b}, impl=impl)
        expected = np.full((512,), i * 2.0 + 2 * i, dtype=np.float32)
        np.testing.assert_allclose(_to_np(out["c"]), expected, rtol=1e-6)


def test_a_and_b_agree(gpu_tesseract):
    """The two strategies must produce identical results."""
    from tesseract_jax_gpu.frontend import apply_gpu_direct

    a = jnp.arange(4096, dtype=jnp.float32) * 0.01
    b = jnp.cos(jnp.arange(4096, dtype=jnp.float32))
    out_a = apply_gpu_direct(gpu_tesseract, {"a": a, "b": b}, impl="A")
    out_b = apply_gpu_direct(gpu_tesseract, {"a": a, "b": b}, impl="B")
    np.testing.assert_array_equal(_to_np(out_a["c"]), _to_np(out_b["c"]))
