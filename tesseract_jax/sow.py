# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sow and save_intermediates mechanism for capturing intermediate values and gradients.

This module provides ``sow`` and ``save_intermediates``, a pair of functions for tagging
and extracting intermediate values (primals, tangents, cotangents) during
JAX computations.  This is useful for debugging pipelines that chain
multiple :func:`~tesseract_jax.apply_tesseract` calls.

Example::

    from tesseract_jax import apply_tesseract, sow, save_intermediates


    def my_pipeline(inputs):
        res = apply_tesseract(tess1, inputs)
        res = sow(res, "step1")
        res = apply_tesseract(tess2, res)
        return res["output"].sum()


    # Forward only — capture primals
    result, intermediates = save_intermediates(my_pipeline)(inputs)

    # With grad — capture primals + cotangents
    grads, intermediates = save_intermediates(jax.grad(my_pipeline))(inputs)
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, TypeVar

import jax
import jax.numpy as jnp
import jax.tree_util
from jax import extend
from jax.interpreters import ad, batching, mlir
from jax.typing import ArrayLike

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Primitive definition
# ---------------------------------------------------------------------------

sow_p = extend.core.Primitive("sow")
sow_p.multiple_results = True


def _sow_impl(*args: Any, name: str, tag: str, mode: str) -> tuple:
    """Implement the sow primitive as a no-op identity."""
    return args


sow_p.def_impl(_sow_impl)


@sow_p.def_abstract_eval
def _sow_abstract_eval(*args: Any, name: str, tag: str, mode: str) -> tuple:
    """Return abstract values unchanged."""
    return args


def _sow_lowering(ctx: Any, *args: Any, name: str, tag: str, mode: str) -> tuple:
    """Lower sow to MLIR as identity."""
    return args


mlir.register_lowering(sow_p, _sow_lowering)


# NOTE: The JVP, transpose, and batching rules below intentionally omit
# type annotations to avoid runtime type-checking by typeguard (which
# is active in the test suite).  JAX's internal dispatchers pass list
# objects where tuple annotations would cause TypeCheckError.


def _sow_jvp(
    in_args: tuple[ArrayLike, ...],
    tan_args: tuple[ArrayLike, ...],
    *,
    name: str,
    tag: str,
    mode: str,
) -> tuple[tuple[ArrayLike, ...], tuple[ArrayLike, ...]]:
    """JVP rule: sow primals and tangents under separate modes."""
    primals = sow_p.bind(*in_args, name=name, tag=tag, mode="primal")
    # Instantiate symbolic zeros to concrete zeros
    tan_args_concrete = tuple(
        jnp.zeros_like(p) if isinstance(t, jax._src.ad_util.Zero) else t
        for p, t in zip(in_args, tan_args, strict=True)
    )
    tangents = sow_p.bind(*tan_args_concrete, name=name, tag=tag, mode="tangent")
    return tuple(primals), tuple(tangents)


ad.primitive_jvps[sow_p] = _sow_jvp


def _sow_transpose(
    cotangents: Sequence[ArrayLike],
    *args: Any,
    name: str,
    tag: str,
    mode: str,
) -> Sequence[ArrayLike]:
    """Transpose rule: sow cotangents and pass them through."""
    cotan_concrete = tuple(
        jnp.zeros_like(a.aval) if isinstance(a, jax._src.ad_util.Zero) else a
        for a in cotangents
    )
    sow_p.bind(*cotan_concrete, name=name, tag=tag, mode="cotangent")
    return cotangents


ad.primitive_transposes[sow_p] = _sow_transpose


def _sow_batching(
    args: Sequence[ArrayLike],
    axes: Sequence[Any],
    *,
    name: str,
    tag: str,
    mode: str,
) -> tuple[Sequence[ArrayLike], Sequence[Any]]:
    """Batching rule: pass batched args through unchanged."""
    return sow_p.bind(*args, name=name, tag=tag, mode=mode), axes


batching.primitive_batchers[sow_p] = _sow_batching

# ---------------------------------------------------------------------------
# Treedef storage (populated at trace time, read by ``save_intermediates``)
# ---------------------------------------------------------------------------

_sow_treedefs: dict[str, jax.tree_util.PyTreeDef] = {}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def sow(value: T, name: str, *, tag: str = "intermediates") -> T:
    """Tag an intermediate value for capture by :func:`save_intermediates`.

    Acts as the identity function: the return value is always equal to
    *value*.  When the enclosing function is later wrapped with
    :func:`save_intermediates`, the tagged value (and, if a derivative transformation
    is active, its tangent or cotangent) will appear in the returned
    intermediates dictionary.

    Args:
        value: Any JAX-compatible pytree (dict, list, array, nested
            combinations, ...).
        name: A unique string name used to identify this intermediate in
            the dictionary returned by :func:`save_intermediates`.  Using the same
            *name* twice inside a single function raises ``ValueError``
            at save_intermediates time.
        tag: An optional string tag for grouping intermediates.  Only
            intermediates whose tag matches the one passed to
            :func:`save_intermediates` will be captured.  Defaults to
            ``"intermediates"``.

    Returns:
        *value*, unchanged.
    """
    flat, treedef = jax.tree.flatten(value)
    _sow_treedefs[name] = treedef
    results = sow_p.bind(*flat, name=name, tag=tag, mode="primal")
    return jax.tree.unflatten(treedef, results)


def _rewrite_jaxpr(
    jaxpr: extend.core.Jaxpr,
    tag: str,
) -> tuple:
    """Recursively rewrite a jaxpr to surface sow values as extra outputs.

    Walks all equations in *jaxpr*.  For each ``sow_p`` with matching *tag*,
    its input vars are appended as additional output vars.  For equations
    that contain sub-jaxprs (e.g. ``pjit``), the rewriting recurses into
    the sub-jaxpr, and new output variables are created to thread the
    extra values up to the parent.

    Returns:
        A tuple ``(new_jaxpr, sow_keys, sow_counts)`` where *sow_keys*
        is a list of ``(name, mode)`` pairs and *sow_counts* gives the
        number of flat values for each key.
    """
    new_eqns: list = []
    extra_outvars: list = []
    sow_keys: list[tuple[str, str]] = []
    sow_counts: list[int] = []

    for eqn in jaxpr.eqns:
        # Look for a sub-jaxpr (pjit, while_loop, cond, etc.)
        sub_jaxpr_key = None
        for pk, pv in eqn.params.items():
            if hasattr(pv, "jaxpr") and hasattr(pv.jaxpr, "eqns"):
                sub_jaxpr_key = pk
                break

        if sub_jaxpr_key is not None:
            sub_closed = eqn.params[sub_jaxpr_key]
            new_sub_jaxpr, sub_keys, sub_counts = _rewrite_jaxpr(sub_closed.jaxpr, tag)

            if sub_keys:
                new_sub_closed = sub_closed.replace(jaxpr=new_sub_jaxpr)
                new_params = dict(eqn.params)
                new_params[sub_jaxpr_key] = new_sub_closed

                n_extra = len(new_sub_jaxpr.outvars) - len(sub_closed.jaxpr.outvars)
                if "out_shardings" in new_params:
                    orig = new_params["out_shardings"]
                    new_params["out_shardings"] = tuple(orig) + (orig[0],) * n_extra
                if "out_layouts" in new_params:
                    orig = new_params["out_layouts"]
                    new_params["out_layouts"] = tuple(orig) + (None,) * n_extra

                extra_avals = [
                    v.aval
                    for v in new_sub_jaxpr.outvars[len(sub_closed.jaxpr.outvars) :]
                ]
                new_out_vars = list(eqn.outvars)
                for aval in extra_avals:
                    new_var = extend.core.Var("", aval)
                    new_out_vars.append(new_var)
                    extra_outvars.append(new_var)

                sow_keys.extend(sub_keys)
                sow_counts.extend(sub_counts)
                new_eqns.append(eqn.replace(params=new_params, outvars=new_out_vars))
            else:
                new_eqns.append(eqn)
        elif eqn.primitive is sow_p and eqn.params.get("tag") == tag:
            new_eqns.append(eqn)
            name = eqn.params["name"]
            mode = eqn.params["mode"]
            sow_keys.append((name, mode))
            sow_counts.append(len(eqn.invars))
            extra_outvars.extend(eqn.invars)
        else:
            new_eqns.append(eqn)

    if extra_outvars:
        new_jaxpr = jaxpr.replace(
            eqns=new_eqns,
            outvars=list(jaxpr.outvars) + extra_outvars,
        )
        return new_jaxpr, sow_keys, sow_counts
    return jaxpr, sow_keys, sow_counts


def _validate_no_duplicate_sow_names(
    sow_keys: list[tuple[str, str]],
) -> None:
    """Raise ValueError if a sow name appears more than once for the same mode."""
    seen: dict[str, str] = {}
    for name, mode in sow_keys:
        if name in seen and seen[name] == mode:
            raise ValueError(
                f"Duplicate sow name {name!r} (mode={mode!r}). "
                "Each sow name must be unique within a single function."
            )
        seen.setdefault(name, mode)


def save_intermediates(
    fn: Callable[..., Any], *, tag: str = "intermediates"
) -> Callable[..., tuple[Any, dict[str, dict[str, Any]]]]:
    """Functional transformation that captures values tagged by :func:`sow`.

    Returns a new function with the same signature as *fn*, but whose
    return value is a tuple ``(original_result, intermediates)`` where
    *intermediates* is a dictionary mapping sow names to sub-dictionaries
    with keys ``"primal"``, ``"tangent"``, and/or ``"cotangent"``.

    Which keys are present depends on the JAX transformations applied to
    *fn* before wrapping with ``save_intermediates``:

    * Plain call: only ``"primal"``
    * ``jax.grad`` / ``jax.vjp``: ``"primal"`` and ``"cotangent"``
    * ``jax.jvp``: ``"primal"`` and ``"tangent"``

    ``save_intermediates`` should be the **outermost** transformation.
    It recursively descends into sub-jaxprs (e.g. inside ``jax.jit``
    boundaries) so ``sow`` calls inside JIT-compiled functions are
    captured correctly.

    Args:
        fn: The function to wrap.
        tag: Only capture intermediates whose tag matches this string.
            Defaults to ``"intermediates"``.

    Returns:
        A new callable ``(*args, **kwargs) -> (result, intermediates)``.

    Raises:
        ValueError: If a sow *name* is used more than once inside *fn*.
    """

    def wrapped(*args: Any, **kwargs: Any) -> tuple[Any, dict[str, dict[str, Any]]]:
        _sow_treedefs.clear()

        closed_jaxpr = jax.make_jaxpr(lambda *a: fn(*a, **kwargs))(*args)
        jaxpr = closed_jaxpr.jaxpr
        treedefs = dict(_sow_treedefs)

        # Recursively rewrite the jaxpr to surface sow values
        new_jaxpr, sow_keys, sow_counts = _rewrite_jaxpr(jaxpr, tag)
        _validate_no_duplicate_sow_names(sow_keys)

        if not sow_keys:
            result = jax.core.eval_jaxpr(jaxpr, closed_jaxpr.consts, *args)
            result = result[0] if len(result) == 1 else tuple(result)
            return result, {}

        all_results = jax.core.eval_jaxpr(new_jaxpr, closed_jaxpr.consts, *args)

        n_original = len(jaxpr.outvars)
        original_results = all_results[:n_original]
        extra_results = all_results[n_original:]

        # Group by (name, mode) and reconstruct pytrees
        intermediates: dict[str, dict[str, Any]] = {}
        idx = 0
        for key, count in zip(sow_keys, sow_counts, strict=True):
            name, mode = key
            vals = extra_results[idx : idx + count]
            treedef = treedefs.get(name)
            if treedef is not None:
                value = jax.tree.unflatten(treedef, vals)
            else:
                value = vals[0] if count == 1 else tuple(vals)
            intermediates.setdefault(name, {})[mode] = value
            idx += count

        result = (
            original_results[0]
            if len(original_results) == 1
            else tuple(original_results)
        )
        return result, intermediates

    return wrapped
