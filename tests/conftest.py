# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import contextlib
import importlib.util
import os
import sys
from pathlib import Path

import jax
import numpy as np
import pytest
from tesseract_core import Tesseract

here = Path(__file__).parent

jax.config.update("jax_enable_x64", True)

# Make the native FFI handler reject any buffer that crosses its boundary in host
# memory, so every GPU-direct test also guards against a host copy sneaking onto a
# dispatch path. The shim reads this once, on its first dispatch, so it is set
# before any test runs.
os.environ["TESSERACT_JAX_DEBUG_CHECK_DEVICE_PTRS"] = "1"


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires a CUDA GPU")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strip_functions_from_api(source: str, func_names: set[str]) -> str:
    """Return *source* with top-level function definitions in *func_names* removed."""
    tree = ast.parse(source)
    # Collect line ranges (1-indexed) of functions to remove
    remove_ranges: list[tuple[int, int]] = []
    for node in ast.iter_child_nodes(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in func_names
            and node.end_lineno is not None
        ):
            remove_ranges.append((node.lineno, node.end_lineno))

    if not remove_ranges:
        return source

    lines = source.splitlines(keepends=True)
    keep: list[str] = []
    for i, line in enumerate(lines, start=1):
        if not any(start <= i <= end for start, end in remove_ranges):
            keep.append(line)
    return "".join(keep)


def _serve_tesseract(
    api_path: str | Path, *, env: dict[str, str] | None = None, **kwargs
):
    """Yield a client for a test Tesseract served in a subprocess on this interpreter.

    The subprocess shares the test environment's packages but not its JAX runtime.
    It inherits this process's environment, plus ``env`` while it is spawned.
    """
    with contextlib.ExitStack() as stack:
        with pytest.MonkeyPatch.context() as mp:
            for key, value in (env or {}).items():
                mp.setenv(key, value)
            tess = stack.enter_context(
                Tesseract.from_source(
                    api_path, python_executable=sys.executable, **kwargs
                )
            )
        yield tess


def _load_tesseract(folder_name: str) -> Tesseract:
    """Load a Tesseract directly from a test API file."""
    return Tesseract.from_tesseract_api(f"tests/{folder_name}/tesseract_api.py")


# ---------------------------------------------------------------------------
# GPU (cuda_ipc) serving
# ---------------------------------------------------------------------------
#
# Cross-process CUDA IPC needs the Tesseract (producer) and the test process
# (consumer) to be *separate* processes sharing the GPU -- a process cannot open
# an IPC handle it exported itself. ``from_source`` provides that separate process.
#
# The GPU Tesseracts compute with JAX, the main use case. The tests whose outcome
# depends on the producer also run against CuPy, standing in for any other GPU
# framework. The two differ in how they adopt device inputs (DLPack for JAX,
# ``__cuda_array_interface__`` for CuPy) and in which allocator their outputs
# come from.
#
# The GPU tests skip without a GPU, and the CuPy-producer ones also without CuPy,
# unless TESSERACT_JAX_GPU_REQUIRED is set (as in the GPU CI job), in which case
# they fail instead.

GPU_PRODUCERS = ["jax", "cupy"]


def _gpu_available() -> bool:
    try:
        return any(d.platform == "gpu" for d in jax.devices())
    except Exception:  # noqa: BLE001 - probing for a GPU must never raise
        return False


def _skip_or_fail(reason: str) -> None:
    if os.environ.get("TESSERACT_JAX_GPU_REQUIRED"):
        pytest.fail(f"{reason}, but TESSERACT_JAX_GPU_REQUIRED is set")
    pytest.skip(reason)


def _require_gpu_producer(producer: str) -> None:
    if not _gpu_available():
        _skip_or_fail("no GPU backend for JAX")
    if producer == "cupy" and importlib.util.find_spec("cupy") is None:
        _skip_or_fail("CuPy is not installed")


def _served_gpu_tesseract(folder: str, producer: str = "jax"):
    """Skip-or-serve helper shared by the GPU Tesseract fixtures."""
    _require_gpu_producer(producer)
    yield from _serve_tesseract(
        here / folder / "tesseract_api.py",
        env={"TESSERACT_JAX_TEST_ARRAY_MODULE": producer},
        # cuda_ipc GPU transport is an experimental opt-in in tesseract-core.
        gpu_transport="cuda_ipc",
    )


@pytest.fixture(scope="module")
def served_gpu_tesseract():
    """A served all-float32 GPU Tesseract computing with JAX. Skips without a GPU."""
    yield from _served_gpu_tesseract("gpu_tesseract")


@pytest.fixture(scope="module", params=GPU_PRODUCERS)
def served_gpu_mixed_dtype_tesseract(request):
    """A served GPU Tesseract with float32 in / float64 out, once per producer."""
    yield from _served_gpu_tesseract("gpu_mixed_dtype_tesseract", request.param)


# ---------------------------------------------------------------------------
# Parametrised transport fixture (host vs cuda_ipc, per producer)
# ---------------------------------------------------------------------------
#
# The platform-sensitive behaviours -- dtype handling, discarded-slot fills,
# non-differentiable inputs/outputs, jacobian fwd/bwd, batching -- must agree
# between the two dispatch lowerings and across producers. Rather than duplicate
# each test, this fixture serves the *array-module-agnostic*
# ``transport_tesseract`` in one of three modes and yields the client together
# with the ``apply_tesseract`` kwargs that select the transport, so a single test
# body runs on all of them:
#
#   * "host"          -> no GPU transport: numpy compute, device->host->device
#   * "cuda_ipc-jax"  -> cuda_ipc GPU transport: JAX compute
#   * "cuda_ipc-cupy" -> cuda_ipc GPU transport: CuPy compute
#
# The cuda_ipc legs skip (or fail) like the standalone GPU fixtures above.


@pytest.fixture(
    params=[
        "host",
        # The cuda_ipc legs carry the ``gpu`` marker so they are collected under
        # ``-m gpu`` (the GPU CI job) and excluded from the CPU job, while the host
        # leg runs everywhere. All legs share one test body.
        *(
            pytest.param(f"cuda_ipc-{producer}", marks=pytest.mark.gpu)
            for producer in GPU_PRODUCERS
        ),
    ]
)
def transport(request):
    """Yield ``(client, apply_kwargs)`` for one dispatch transport and producer."""
    if request.param == "host":
        for client in _serve_tesseract(
            here / "transport_tesseract" / "tesseract_api.py"
        ):
            yield client, {}
    else:
        producer = request.param.removeprefix("cuda_ipc-")
        for client in _served_gpu_tesseract("transport_tesseract", producer):
            yield client, {"gpu_transport": "cuda_ipc"}


# ---------------------------------------------------------------------------
# Served fixtures  (session-scoped, each serves a Tesseract in its own process)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def served_univariate_tesseract():
    yield from _serve_tesseract(here / "univariate_tesseract" / "tesseract_api.py")


@pytest.fixture(scope="session")
def served_nested_tesseract():
    yield from _serve_tesseract(here / "nested_tesseract" / "tesseract_api.py")


@pytest.fixture(scope="session")
def served_vectoradd_tesseract():
    yield from _serve_tesseract(here / "vectoradd_tesseract" / "tesseract_api.py")


@pytest.fixture(scope="session")
def served_pytree_tesseract():
    yield from _serve_tesseract(here / "pytree_tesseract" / "tesseract_api.py")


@pytest.fixture(scope="session")
def served_batched_tesseract():
    yield from _serve_tesseract(here / "batched_tesseract" / "tesseract_api.py")


# Tesseracts with specific endpoints removed — generated dynamically from
# the base univariate_tesseract so we don't need separate directories.


@pytest.fixture(scope="session")
def served_tesseract_no_jvp(tmp_path_factory):
    source = (here / "univariate_tesseract" / "tesseract_api.py").read_text()
    stripped = _strip_functions_from_api(source, {"jacobian_vector_product"})
    api_file = tmp_path_factory.mktemp("univariate_no_jvp") / "tesseract_api.py"
    api_file.write_text(stripped)
    yield from _serve_tesseract(api_file)


@pytest.fixture(scope="session")
def served_tesseract_no_vjp(tmp_path_factory):
    source = (here / "univariate_tesseract" / "tesseract_api.py").read_text()
    stripped = _strip_functions_from_api(source, {"vector_jacobian_product"})
    api_file = tmp_path_factory.mktemp("univariate_no_vjp") / "tesseract_api.py"
    api_file.write_text(stripped)
    yield from _serve_tesseract(api_file)


# ---------------------------------------------------------------------------
# Direct-load fixtures  (function-scoped, no server needed)
# ---------------------------------------------------------------------------


@pytest.fixture
def pytree_tess() -> Tesseract:
    return _load_tesseract("pytree_tesseract")


@pytest.fixture
def dict_key_tess() -> Tesseract:
    return _load_tesseract("dict_key_tesseract")


@pytest.fixture
def univariate_tess() -> Tesseract:
    return _load_tesseract("univariate_tesseract")


@pytest.fixture
def batched_tess() -> Tesseract:
    """Ellipsis-shaped schema, so the vectorized vmap methods are legal here."""
    return _load_tesseract("batched_tesseract")


@pytest.fixture
def vectoradd_tess() -> Tesseract:
    return _load_tesseract("vectoradd_tesseract")


@pytest.fixture
def static_input_tess() -> Tesseract:
    return _load_tesseract("static_input_tesseract")


@pytest.fixture
def mixed_dtype_tess() -> Tesseract:
    return _load_tesseract("mixed_dtype_tesseract")


@pytest.fixture
def gather_tess() -> Tesseract:
    return _load_tesseract("gather_tesseract")


@pytest.fixture
def validating_tess() -> Tesseract:
    return _load_tesseract("validating_tesseract")


@pytest.fixture
def nonarray_output_tess() -> Tesseract:
    """OutputSchema mixes real arrays with a str and a bool."""
    return _load_tesseract("nonarray_output_tesseract")


@pytest.fixture
def non_abstract_tess() -> Tesseract:
    """No abstract_eval endpoint, so a JAX transformation has to be rejected."""
    return _load_tesseract("non_abstract_tesseract")


@pytest.fixture
def drifting_static_tess() -> Tesseract:
    """Its apply reports a static output that abstract_eval did not predict."""
    return _load_tesseract("drifting_static_tesseract")


@pytest.fixture
def zero_cotangent_tess() -> Tesseract:
    """Two differentiable outputs, one with a NaN gradient at x = 0."""
    return _load_tesseract("zero_cotangent_tesseract")


# ---------------------------------------------------------------------------
# Shared test inputs
# ---------------------------------------------------------------------------


@pytest.fixture
def pytree_tess_inputs() -> dict:
    """Provide inputs for pytree_tesseract tests with different shapes."""
    x = np.array([1.0, 2.0, 3.0], dtype="float32")  # shape (3,)
    y = np.array([4.0, 5.0, 6.0, 7.0], dtype="float32")  # shape (4,)
    z = np.array([8.0, 9.0, 10.0, 11.0, 12.0], dtype="float32")  # shape (5,)
    u = np.array([13.0, 14.0, 15.0, 16.0, 17.0, 18.0], dtype="float32")  # shape (6,)
    v = np.array(
        [19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0], dtype="float32"
    )  # shape (7,)
    d0 = np.array(
        [26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0], dtype="float32"
    )  # shape (8,)
    d1 = np.array(
        [34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0], dtype="float32"
    )  # shape (9,)
    k = np.array([2.0, 2.0], dtype="float32")  # shape (2,) non-differentiable
    m = np.array(
        [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0], dtype="float32"
    )  # shape (10,) non-differentiable
    z0 = np.array(
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype="float32"
    )  # shape (11,) non-differentiable
    z1 = np.array(
        [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0], dtype="float32"
    )  # shape (12,) non-differentiable

    return {
        "alpha": {
            "x": x,
            "y": y,
        },
        "beta": {"z": z, "gamma": {"u": u, "v": v}},
        "delta": [d0, d1],
        "epsilon": {"k": k, "m": m},
        "zeta": [z0, z1],
    }
