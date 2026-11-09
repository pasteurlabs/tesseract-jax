# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check the gradient of a function that calls Tesseracts, one Tesseract at a time.

:func:`check_gradients` compares the gradient of a loss with a central finite
difference along a random direction. While it computes that gradient, it
records the inputs that reach each ``apply`` call, and then runs the same check
on each Tesseract alone at those inputs. That tells a wrong Tesseract apart
from wrong JAX code between the Tesseracts.
"""

import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from tesseract_core import Tesseract

from tesseract_jax.dispatch_params import DispatchParams
from tesseract_jax.primitive import apply_tesseract, record_apply_inputs
from tesseract_jax.tesseract_compat import to_host
from tesseract_jax.tree_util import (
    PyTree,
    pytree_to_path_dict,
    split_args,
    unflatten_args,
)

# The title ``Tesseract.from_tesseract_api`` gives a Tesseract without a config.
_DEFAULT_TITLE = "Tesseract"

_SKIPPED = "no differentiable inputs, skipped"
_UNSURE = "finite difference too uncertain"


@dataclass(frozen=True)
class Check:
    """The directional derivative check of one function.

    Attributes:
        name: ``"pipeline"``, or the label of a Tesseract call.
        grad_dot_d: The gradient contracted with the random direction.
        steps: The step sizes tried, relative to each input's RMS.
        finite_differences: The central difference at each step, NaN where the
            function raised or returned a non-finite value.
        fd: The finite difference with the smallest error bound (see
            :func:`check_gradients`).
        fd_bound: That bound: how far ``fd`` may be from the true derivative.
        step: The step that gave ``fd``.
        rel_error: ``|grad_dot_d - fd| / max(|grad_dot_d|, |fd|)``.
        problems: NaN or inf in the gradient, a finite difference too uncertain
            to tell whether the gradient is right, or why the check was
            skipped.
        error: The exception raised while computing the gradient, if any.
        passed: Whether there is no error and no problem, and the gradient
            agrees within ``rtol`` with a finite difference, after allowing
            for its error bound.
    """

    name: str
    grad_dot_d: float
    steps: tuple[float, ...]
    finite_differences: tuple[float, ...]
    fd: float
    fd_bound: float
    step: float
    rel_error: float
    problems: tuple[str, ...]
    error: str | None
    passed: bool


@dataclass(frozen=True)
class CallCheck(Check):
    """The check of one Tesseract call, at the inputs it received.

    Attributes:
        tesseract: The Tesseract that was called.
        inputs: The inputs it received, as NumPy arrays.
        site: The call site, numbered from 1 in the order the sites first ran.
            Only the first execution of each site is checked.
        n_calls: How many times the site ran in the forward pass.
    """

    tesseract: Tesseract
    inputs: PyTree
    site: int
    n_calls: int


@dataclass(frozen=True)
class GradientReport:
    """The result of :func:`check_gradients`.

    Attributes:
        pipeline: The check of ``loss_fn`` as a whole.
        calls: One check per Tesseract call site, at the inputs of its first
            execution, in the order the sites first ran.
        rtol: The relative tolerance the checks used.
        seed: The seed of the random directions.
    """

    pipeline: Check
    calls: tuple[CallCheck, ...]
    rtol: float
    seed: int

    @property
    def passed(self) -> bool:
        """Whether the pipeline and every Tesseract call passed."""
        return self.pipeline.passed and all(c.passed for c in self.calls)

    @property
    def verdict(self) -> str:
        """A few sentences on where to look first."""
        failing = [c for c in self.calls if not c.passed]
        if failing:
            # A row that fails for sure says more than one that cannot tell.
            first = next((c for c in failing if not _unsure(c)), failing[0])
            if first.error is not None:
                return (
                    f"{first.name} raised when checked on its own, at the inputs it "
                    "received in the pipeline; its row shows the error. Start there."
                )
            verdict = (
                f"{first.name} disagrees with a finite difference of its own "
                "apply, at the inputs it received in the pipeline. Start there."
            )
            if _unsure(first):
                verdict += (
                    " Its finite difference is too uncertain to be sure, as with a "
                    "noisy output or in float32, so the gradient may still be "
                    "right; check at another point too."
                )
            if self.pipeline.passed:
                verdict += (
                    " The pipeline gradient still agrees along this direction, so "
                    "the error may not reach this loss; try another seed."
                )
            return verdict
        if self.pipeline.error is not None:
            return (
                "The gradient of loss_fn could not be computed, so the pipeline "
                "was not checked; its row shows the error."
            )
        if _unsure(self.pipeline):
            return (
                "The finite difference of the pipeline is too uncertain, so this "
                "check cannot tell whether its gradient is wrong. This happens "
                "near a minimum of the loss, with a noisy loss, or in float32; "
                "check at another point, or with jax_enable_x64."
            )
        if not self.calls:
            agrees = "agrees" if self.pipeline.passed else "does not agree"
            return (
                "No Tesseract was called while evaluating loss_fn; only the "
                f"pipeline was checked. The pipeline gradient {agrees}."
            )
        if self.pipeline.passed:
            return "Gradients agree."
        repeated = [
            f"{c.name} ({c.n_calls} calls)" for c in self.calls if c.n_calls > 1
        ]
        if repeated:
            return (
                "No checked Tesseract call disagrees, but the pipeline does. Only "
                f"the first call of each site was checked: {', '.join(repeated)}, "
                "so a later call may be the one that is wrong."
            )
        # Each Tesseract may be off by up to rtol, and that can grow along the
        # pipeline, so the code between them is only blamed when the pipeline
        # is off by far more than any Tesseract.
        closest = max(self.calls, key=lambda c: _error_range(c)[1])
        if _error_range(self.pipeline)[0] <= 10 * _error_range(closest)[1]:
            return (
                "No Tesseract call fails, but the pipeline does, by less than ten "
                f"times the error of {closest.name}, so this check cannot tell "
                "whether that Tesseract or the JAX code between the Tesseracts "
                "is wrong. Rerun with a smaller rtol to tell."
            )
        return (
            "Every Tesseract agrees with finite differences at the inputs it "
            "received, but the pipeline does not. The problem is in the JAX code "
            "between the Tesseracts: look for jax.lax.stop_gradient, operations "
            "with zero or undefined derivatives (round, argmax, integer casts, "
            "comparisons, sqrt at 0), or a custom_vjp."
        )

    def __repr__(self) -> str:
        steps = self.pipeline.steps
        width = max(len(c.name) for c in (self.pipeline, *self.calls))
        lines = [
            (
                f"Gradient check along a random direction (seed={self.seed}, "
                f"rtol={self.rtol}, steps {steps[0]:.0e} .. {steps[-1]:.0e})"
            ),
            "",
            (
                f"  {'':<{width}}  {'grad.d':>10}  {'fin. diff':>10}  "
                f"{'rel. error':>10}  {'step':>6}  result"
            ),
            _format_row(self.pipeline, width),
        ]
        if self.calls:
            lines.append("")
        lines += [_format_row(c, width) for c in self.calls]
        lines += ["", self.verdict]
        return "\n".join(lines)


def _unsure(check: Check) -> bool:
    return _UNSURE in check.problems


def _error_range(check: Check) -> tuple[float, float]:
    """The least and the most relative error ``check`` allows, given ``fd_bound``."""
    miss = abs(check.grad_dot_d - check.fd)
    scale = max(abs(check.grad_dot_d), abs(check.fd))
    if miss == 0:
        return 0.0, 0.0
    return max(miss - check.fd_bound, 0) / scale, (miss + check.fd_bound) / scale


def _format_row(check: Check, width: int) -> str:
    if check.error is not None:
        result = f"ERROR  ({check.error})"
    else:
        result = "ok" if check.passed else "FAIL"
        if check.problems:
            result += f"  ({'; '.join(check.problems)})"
    return (
        f"  {check.name:<{width}}  {check.grad_dot_d:10.2e}  "
        f"{check.fd:10.2e}  {check.rel_error:10.2e}  {check.step:6.0e}  {result}"
    )


def _site_label(tesseract: Tesseract, site: int) -> str:
    title = tesseract.openapi_schema["info"]["title"]
    if title == _DEFAULT_TITLE:
        schemas = tesseract.openapi_schema["components"]["schemas"]
        inputs = ", ".join(schemas["ApplyInputSchema"]["differentiable_arrays"])
        outputs = ", ".join(schemas["ApplyOutputSchema"]["differentiable_arrays"])
        title = f"{inputs} -> {outputs}"
    return f"#{site} {title}"


def _one_line(exc: Exception) -> str:
    # A host callback wraps the endpoint's exception in a long traceback whose
    # last line is the original exception, so that line says the most.
    lines = [line.strip() for line in str(exc).splitlines() if line.strip()]
    return f"{type(exc).__name__}: {lines[-1] if lines else ''}"


class _Recorder:
    """The inputs of the first ``apply`` call at each call site."""

    def __init__(self) -> None:
        self.inputs: dict[DispatchParams, tuple[np.ndarray, ...]] = {}
        self.n_calls: dict[DispatchParams, int] = {}
        # Host callbacks can run on XLA's own threads.
        self._lock = threading.Lock()

    def record(self, params: DispatchParams, args: tuple) -> None:
        with self._lock:
            self.n_calls[params] = self.n_calls.get(params, 0) + 1
            if params not in self.inputs:
                # Copy, so a record never shares memory with a buffer XLA owns.
                self.inputs[params] = tuple(
                    np.array(to_host(a), copy=True) for a in args
                )


def _is_float(leaf: Any) -> bool:
    dtype = getattr(leaf, "dtype", None)
    if dtype is None:
        dtype = np.result_type(leaf)
    return jnp.issubdtype(dtype, jnp.floating)


def _scaled_direction(key: jax.Array, x: dict[str, jax.Array]) -> dict[str, np.ndarray]:
    """A standard normal direction, scaled by the RMS of each input."""
    direction = {}
    for k, (name, leaf) in zip(jax.random.split(key, len(x)), x.items(), strict=True):
        leaf = np.asarray(leaf)
        rms = np.sqrt(np.mean(np.square(leaf, dtype=np.float64))) if leaf.size else 0
        z = np.asarray(jax.random.normal(k, leaf.shape, leaf.dtype))
        direction[name] = np.asarray(z * (rms or 1.0), dtype=leaf.dtype)
    return direction


def _rel_error(g: float, fd: float) -> float:
    if not (np.isfinite(g) and np.isfinite(fd)):
        return np.inf
    if g == fd == 0:
        return 0.0
    return abs(g - fd) / max(abs(g), abs(fd))


def _directional_check(
    f: Callable[[dict[str, jax.Array]], tuple[float, float]],
    x: dict[str, jax.Array],
    grad: dict[str, jax.Array] | None,
    error: str | None,
    key: jax.Array,
    steps: tuple[float, ...],
    rtol: float,
) -> dict[str, Any]:
    """Compare ``grad . d`` with central differences of ``f`` along ``d``.

    ``f`` returns a value and how much rounding it to its precision can change
    it. Returns the fields of a :class:`Check`, except ``name``.
    """
    if not x:
        return dict(
            grad_dot_d=0.0,
            steps=steps,
            finite_differences=(0.0,) * len(steps),
            fd=0.0,
            fd_bound=0.0,
            step=steps[0],
            rel_error=0.0,
            problems=(_SKIPPED,),
            error=error,
            passed=error is None,
        )

    d = _scaled_direction(key, x)
    fds, rounding = [], []
    for h in steps:
        try:
            plus, rounding_plus = f({k: x[k] + h * d[k] for k in x})
            minus, rounding_minus = f({k: x[k] - h * d[k] for k in x})
            fd = (plus - minus) / (2 * h)
            # How far rounding f to its precision can move this difference.
            rounded = (rounding_plus + rounding_minus) / (2 * h)
        except Exception:  # noqa: BLE001 - e.g. a step that leaves the valid domain
            fd = rounded = np.nan
        fds.append(fd if np.isfinite(fd) else np.nan)
        rounding.append(rounded)

    problems = []
    if grad is None:
        g = np.nan
    else:
        gradient = {k: np.asarray(grad[k], dtype=np.float64) for k in x}
        g = float(sum(np.vdot(gradient[k], d[k].astype(np.float64)) for k in x))
        nan_paths = [k for k in x if np.isnan(gradient[k]).any()]
        inf_paths = [k for k in x if np.isinf(gradient[k]).any()]
        if nan_paths:
            problems.append(f"NaN in gradient of {nan_paths}")
        if inf_paths:
            problems.append(f"inf in gradient of {inf_paths}")

    # A finite difference is off by a curvature error, which shrinks with the
    # step, at least in proportion to it, and by noise and rounding, which
    # grow as it shrinks. So the move to the next coarser step bounds the
    # first (from the next finer step for the coarsest), the move to the next
    # finer step bounds the second, and rounding is added; the finest step has
    # nothing finer and gets no bound. The gradient agrees if it is within
    # rtol of a finite difference after allowing for its bound. It disagrees
    # only if three neighbouring finite differences have settled within rtol
    # of each other, and the gradient is farther than rtol from the middle one
    # by more than that spread. Otherwise, or if both hold, the check cannot
    # tell.
    agrees = disagrees = False
    fd, bound, step = np.nan, np.inf, steps[0]
    for k in range(len(steps) - 1):
        finer = abs(fds[k] - fds[k + 1])
        if k == 0:
            coarser = np.nan
            curvature = finer / (1 - steps[1] / steps[0])
        else:
            coarser = abs(fds[k] - fds[k - 1])
            curvature = coarser / (steps[k - 1] / steps[k] - 1)
        error_bound = curvature + finer + rounding[k]
        if not np.isfinite(error_bound):
            continue
        miss, tol = abs(g - fds[k]), rtol * max(abs(g), abs(fds[k]))
        agrees |= miss + error_bound <= tol
        spread = max(coarser, finer) + rounding[k]
        disagrees |= spread <= rtol * abs(fds[k]) and miss - spread > tol
        if error_bound < bound:
            fd, bound, step = fds[k], error_bound, steps[k]
    # A function that did not move at any step agrees with a zero gradient.
    if g == 0 and not any(fds):
        agrees, disagrees = True, False
    if agrees and disagrees:
        agrees = disagrees = False
    if error is None and not problems and not agrees and not disagrees:
        problems.append(_UNSURE)
    return dict(
        grad_dot_d=g,
        steps=steps,
        finite_differences=tuple(fds),
        fd=fd,
        fd_bound=bound,
        step=step,
        rel_error=_rel_error(g, fd),
        problems=tuple(problems),
        error=error,
        passed=error is None and not problems and agrees,
    )


def _check_call(
    params: DispatchParams,
    args: tuple[np.ndarray, ...],
    live: tuple[bool, ...],
    *,
    site: int,
    n_calls: int,
    key: jax.Array,
    steps: tuple[float, ...],
    rtol: float,
) -> CallCheck:
    """Check one recorded Tesseract call on its own, at the inputs it received.

    Only the inputs flagged in ``live``, those that depend on the pipeline's
    parameters, are perturbed, so constants such as a mesh stay as they were.
    """
    tesseract = params.client.client
    inputs = unflatten_args(
        args, params.static_args, params.input_pytreedef, params.static_input_mask
    )
    leaves, treedef = jax.tree.flatten(inputs)
    paths = pytree_to_path_dict(
        inputs, schema_paths=params.client.differentiable_input_paths
    )
    non_static, _ = split_args(list(paths.items()), params.static_input_mask)
    x = {
        path: jnp.asarray(leaf)
        for (path, leaf), is_live in zip(non_static, live, strict=True)
        if is_live and leaf is not None and _is_float(leaf)
    }

    def outputs(x: dict[str, jax.Array]) -> dict[str, jax.Array]:
        rebuilt = jax.tree.unflatten(
            treedef,
            [x.get(path, leaf) for path, leaf in zip(paths, leaves, strict=True)],
        )
        out = apply_tesseract(
            tesseract, rebuilt, vmap_method=None, check_static_outputs=False
        )
        flat = pytree_to_path_dict(
            out, schema_paths=params.client.differentiable_output_paths
        )
        return {
            path: v
            for path, v in flat.items()
            if isinstance(v, jax.Array) and jnp.issubdtype(v.dtype, jnp.floating)
        }

    # Pull a random cotangent back through the Tesseract's VJP.
    direction_key, cotangent_key = jax.random.split(key)
    grad, error, cotangent = None, None, {}
    if x:
        try:
            out, vjp = jax.vjp(outputs, x)
            keys = jax.random.split(cotangent_key, len(out))
            cotangent = {
                path: jax.random.normal(k, v.shape, v.dtype)
                for k, (path, v) in zip(keys, out.items(), strict=True)
            }
            (grad,) = vjp(cotangent)
        except Exception as exc:  # noqa: BLE001 - reported on the row
            error = _one_line(exc)

    def weighted_sum(x: dict[str, jax.Array]) -> tuple[float, float]:
        # The outputs weighted by the cotangent, whose gradient is ``grad``.
        # Summed in float64, so only the rounding of the outputs themselves
        # counts.
        value = rounding = 0.0
        for path, y in outputs(x).items():
            terms = np.asarray(cotangent[path], np.float64) * np.asarray(y, np.float64)
            value += terms.sum()
            rounding += np.finfo(y.dtype).eps * np.abs(terms).sum()
        return float(value), float(rounding)

    return CallCheck(
        name=_site_label(tesseract, site),
        **_directional_check(weighted_sum, x, grad, error, direction_key, steps, rtol),
        tesseract=tesseract,
        inputs=inputs,
        site=site,
        n_calls=n_calls,
    )


def check_gradients(
    loss_fn: Callable[[PyTree], ArrayLike],
    params: PyTree,
    *,
    seed: int = 0,
    rtol: float = 1e-2,
    steps: Sequence[float] = (1e-1, 1e-2, 1e-3, 1e-4),
) -> GradientReport:
    """Check the gradient of ``loss_fn``, and of each Tesseract it calls.

    Compares ``jax.grad(loss_fn)(params)`` along a random direction with a
    central finite difference of ``loss_fn``. While computing that gradient,
    it records the inputs that reach each Tesseract's ``apply`` in the forward
    pass, then checks each Tesseract alone at those inputs in the same way.
    A Tesseract called more than once at the same call site, as in a loop, is
    checked at the inputs of its first call.
    Only the ``apply``, ``abstract_eval`` and ``vector_jacobian_product``
    endpoints are used.

    The finite difference is taken at every step in ``steps``, scaled by the
    RMS of each input. Each one but the finest gets an error bound from how
    far it moves to its neighbouring steps and from what rounding the
    function to its precision can do. A check passes when the gradient is
    within ``rtol`` of a finite difference after allowing for its bound, and
    has no NaN or inf. It fails when three neighbouring finite differences
    settled within ``rtol`` of each other and the gradient is farther than
    ``rtol`` from them by more than their spread. Otherwise it reports that
    the finite difference is too uncertain.

    Example:
        >>> def loss_fn(params):
        ...     y = apply_tesseract(solver, {"x": params["x"]})["y"]
        ...     return jnp.sum(y**2)
        >>> report = check_gradients(loss_fn, {"x": jnp.ones(3)})
        >>> print(report)
        >>> assert report.passed

    Args:
        loss_fn: A function with a real scalar output, as for ``jax.grad``. It
            may be jitted and may use ``scan``, ``cond``, ``vmap`` and so on.
        params: The point at which to check. Only floating-point leaves are
            perturbed; integer and boolean leaves are held fixed.
        seed: Seed for the random directions and cotangents.
        rtol: Relative tolerance between the gradient and the finite difference.
        steps: Three or more step sizes, relative to the RMS of each input.

    Returns:
        A :class:`~tesseract_jax.gradient_check.GradientReport`. Print it for a
        table and a verdict on where to look first.
    """
    # Largest first, so that neighbouring steps are next to each other.
    steps = tuple(sorted((float(h) for h in steps), reverse=True))
    if len(steps) < 3 or not all(0 < h < np.inf for h in steps):
        raise ValueError("steps must be three or more positive, finite numbers")

    path_leaves, treedef = jax.tree_util.tree_flatten_with_path(params)
    names = [f"params{jax.tree_util.keystr(path)}" for path, _ in path_leaves]
    leaves = [leaf for _, leaf in path_leaves]
    x = {
        name: jnp.asarray(leaf)
        for name, leaf in zip(names, leaves, strict=True)
        if _is_float(leaf)
    }

    def loss_of(x: dict[str, jax.Array]) -> ArrayLike:
        flat = [x.get(name, leaf) for name, leaf in zip(names, leaves, strict=True)]
        return loss_fn(jax.tree_util.tree_unflatten(treedef, flat))

    def loss_and_rounding(x: dict[str, jax.Array]) -> tuple[float, float]:
        loss = np.asarray(loss_of(x))
        return float(loss), float(np.finfo(loss.dtype).eps * abs(loss))

    key = jax.random.PRNGKey(seed)

    # Record the forward pass only: under jax.checkpoint the backward pass can
    # run an apply again, and that is not a new call.
    grad, error = None, None
    recorder = _Recorder()
    with record_apply_inputs(recorder.record):
        try:
            loss, pullback = jax.vjp(loss_of, x)
            # Dispatch is asynchronous, so wait for the applies to run before
            # recording stops.
            jax.block_until_ready(loss)
        except Exception as exc:  # noqa: BLE001 - reported on the pipeline row
            error = _one_line(exc)
    if error is None:
        try:
            if np.ndim(loss) or not _is_float(loss):
                raise TypeError("loss_fn must return a real scalar")
            (grad,) = pullback(jnp.ones_like(loss))
        except Exception as exc:  # noqa: BLE001 - reported on the pipeline row
            error = _one_line(exc)

    pipeline = Check(
        name="pipeline",
        **_directional_check(
            loss_and_rounding, x, grad, error, jax.random.fold_in(key, 0), steps, rtol
        ),
    )

    # Only the inputs that depend on params, and so carry a tangent, are
    # perturbed.
    calls = tuple(
        _check_call(
            call_params,
            args,
            call_params.has_tangent,
            site=site,
            n_calls=recorder.n_calls[call_params],
            key=jax.random.fold_in(key, site),
            steps=steps,
            rtol=rtol,
        )
        for site, (call_params, args) in enumerate(recorder.inputs.items(), start=1)
    )
    return GradientReport(pipeline=pipeline, calls=calls, rtol=rtol, seed=seed)
