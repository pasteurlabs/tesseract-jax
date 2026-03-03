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

import contextvars
import inspect
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

import jax
import jax.numpy as jnp
import jax.tree_util
from jax import extend
from jax.interpreters import ad, batching, mlir
from jax.typing import ArrayLike

T = TypeVar("T")

# Var constructor signature changed across JAX versions:
#   JAX <=0.9:  Var(aval)
#   JAX >=0.6:  Var(suffix, aval)
_var_params = list(inspect.signature(extend.core.Var).parameters)


def _make_var(aval: Any):
    """Create a ``jax.extend.core.Var`` portably across JAX versions."""
    if len(_var_params) >= 2 and _var_params[0] == "suffix":
        return extend.core.Var("", aval)
    return extend.core.Var(aval)


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

# Each ``save_intermediates`` call sets a fresh dict on this ContextVar before
# tracing.  ``sow`` writes into the current dict during tracing.  This avoids
# a shared global and makes concurrent / nested ``save_intermediates`` calls
# safe — each one sees only its own treedefs.
_sow_treedefs: contextvars.ContextVar[dict[str, jax.tree_util.PyTreeDef] | None] = (
    contextvars.ContextVar("_sow_treedefs", default=None)
)

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
    store = _sow_treedefs.get()
    if store is not None:
        store[name] = treedef
    results = sow_p.bind(*flat, name=name, tag=tag, mode="primal")
    return jax.tree.unflatten(treedef, results)


def _find_nested_jaxpr_param(
    eqn_params: dict[str, Any],
) -> str | None:
    """Return the param key that holds a nested ClosedJaxpr, or None.

    JAX operations like ``pjit`` (from ``jax.jit``), ``while_loop``, and
    ``cond`` store their body as a *ClosedJaxpr* in the equation's params
    dict.  This helper identifies which param key (if any) holds such a
    nested program so we can recurse into it.
    """
    for param_name, param_value in eqn_params.items():
        if hasattr(param_value, "jaxpr") and hasattr(param_value.jaxpr, "eqns"):
            return param_name
    return None


def _rewrite_jaxpr(
    jaxpr: extend.core.Jaxpr,
    tag: str,
) -> tuple[extend.core.Jaxpr, list[tuple[str, str]], list[int]]:
    """Rewrite a jaxpr so that sow-tagged values appear as extra outputs.

    A *jaxpr* (JAX expression) is JAX's intermediate representation: a
    list of equations (primitive operations), each consuming and producing
    typed variables.  ``save_intermediates`` traces the user's function
    into a jaxpr, then calls this function to modify it before evaluation.

    The rewriting does two things:

    1. **Direct sow equations** --When an equation is a ``sow_p`` call
       whose ``tag`` matches, its *input* variables (i.e. the values
       flowing *through* the sow) are appended to the jaxpr's output
       list.  This makes them available as return values when the jaxpr
       is later evaluated.

    2. **Nested sub-programs** --Some JAX operations (e.g. ``jax.jit``
       → ``pjit``) wrap an inner jaxpr.  We recurse into these, and if
       sow values were found inside, we:

       - Replace the inner jaxpr with the rewritten version (which now
         has extra outputs).
       - Create fresh variables in the *parent* jaxpr to receive those
         extra outputs (and extend ``out_shardings`` / ``out_layouts``
         metadata so JAX doesn't complain).
       - Append those fresh variables to the parent's output list,
         threading the captured values all the way up.

    Args:
        jaxpr: The JAX expression to rewrite.
        tag: Only capture sow calls whose ``tag`` parameter equals this.

    Returns:
        A tuple of:

        - **rewritten_jaxpr** --The (possibly modified) jaxpr with extra
          outputs appended.
        - **found_sow_keys** --A list of ``(name, mode)`` pairs, one per
          captured sow, in the order they appear.
        - **flat_counts** --For each entry in *found_sow_keys*, how many
          flat (leaf) array values it contributes.  A scalar sow has
          count 1; a pytree sow has count = number of leaves.
    """
    rewritten_eqns: list = []
    extra_output_vars: list = []
    found_sow_keys: list[tuple[str, str]] = []
    flat_counts: list[int] = []

    for eqn in jaxpr.eqns:
        nested_jaxpr_param = _find_nested_jaxpr_param(eqn.params)

        if nested_jaxpr_param is not None:
            # --- Recurse into a nested sub-program (e.g. pjit body) ---
            inner_closed_jaxpr = eqn.params[nested_jaxpr_param]
            rewritten_inner, inner_keys, inner_counts = _rewrite_jaxpr(
                inner_closed_jaxpr.jaxpr, tag
            )

            if not inner_keys:
                # No sow found inside — keep the equation unchanged.
                rewritten_eqns.append(eqn)
                continue

            # The inner jaxpr now has additional outputs.  We need to
            # update the enclosing equation to match.
            updated_inner = inner_closed_jaxpr.replace(jaxpr=rewritten_inner)
            updated_params = dict(eqn.params)
            updated_params[nested_jaxpr_param] = updated_inner

            # Some primitives (pjit) carry per-output metadata that
            # must be extended to cover the new outputs.
            n_new_outputs = len(rewritten_inner.outvars) - len(
                inner_closed_jaxpr.jaxpr.outvars
            )
            if "out_shardings" in updated_params:
                orig = updated_params["out_shardings"]
                updated_params["out_shardings"] = (
                    tuple(orig) + (orig[0],) * n_new_outputs
                )
            if "out_layouts" in updated_params:
                orig = updated_params["out_layouts"]
                updated_params["out_layouts"] = tuple(orig) + (None,) * n_new_outputs

            # Create fresh variables in the parent scope to receive
            # each new output from the inner jaxpr.
            new_outputs = rewritten_inner.outvars[
                len(inner_closed_jaxpr.jaxpr.outvars) :
            ]
            new_inner_output_avals = [v.aval for v in new_outputs]
            expanded_outvars = list(eqn.outvars)
            for aval in new_inner_output_avals:
                fresh_var = _make_var(aval)
                expanded_outvars.append(fresh_var)
                extra_output_vars.append(fresh_var)

            found_sow_keys.extend(inner_keys)
            flat_counts.extend(inner_counts)
            rewritten_eqns.append(
                eqn.replace(params=updated_params, outvars=expanded_outvars)
            )

        elif eqn.primitive is sow_p and eqn.params.get("tag") == tag:
            # --- Direct sow equation: capture its input values ---
            rewritten_eqns.append(eqn)
            sow_name = eqn.params["name"]
            sow_mode = eqn.params["mode"]
            found_sow_keys.append((sow_name, sow_mode))
            flat_counts.append(len(eqn.invars))
            # The sow's *inputs* are the values we want to capture
            # (sow is identity, so invars == outvars in value).
            extra_output_vars.extend(eqn.invars)

        else:
            rewritten_eqns.append(eqn)

    if extra_output_vars:
        rewritten_jaxpr = jaxpr.replace(
            eqns=rewritten_eqns,
            outvars=list(jaxpr.outvars) + extra_output_vars,
        )
        return rewritten_jaxpr, found_sow_keys, flat_counts
    return jaxpr, found_sow_keys, flat_counts


def _validate_no_duplicate_sow_names(
    found_sow_keys: list[tuple[str, str]],
) -> None:
    """Raise ValueError if a sow name appears more than once for the same mode."""
    seen: dict[str, str] = {}
    for name, mode in found_sow_keys:
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
        # Give this call its own fresh treedef store.  ``sow`` writes
        # into it during tracing; we snapshot it afterwards.  Using a
        # ContextVar means concurrent / nested ``save_intermediates``
        # calls each get an isolated store.
        token = _sow_treedefs.set({})

        # Step 1: Trace the function into a jaxpr.  This converts the
        # Python function into JAX's intermediate representation -- a flat
        # list of primitive operations with typed variables.  Tracing
        # also populates the treedef store with the pytree structure of
        # each sow call (needed later to unflatten captured flat arrays
        # back into the original pytree shape).
        closed_jaxpr = jax.make_jaxpr(lambda *a: fn(*a, **kwargs))(*args)
        original_jaxpr = closed_jaxpr.jaxpr
        treedefs_snapshot = dict(_sow_treedefs.get())  # type: ignore[arg-type]

        # Restore the previous ContextVar state so we don't leak.
        _sow_treedefs.reset(token)

        # Step 2: Rewrite the jaxpr so that sow-tagged values become
        # additional outputs.  This walks the entire program (including
        # inside jit boundaries) and appends the sow values to the
        # output list.
        rewritten_jaxpr, found_sow_keys, flat_counts = _rewrite_jaxpr(
            original_jaxpr, tag
        )
        _validate_no_duplicate_sow_names(found_sow_keys)

        # No sow calls found -- just evaluate the original program.
        if not found_sow_keys:
            flat_results = jax.core.eval_jaxpr(
                original_jaxpr, closed_jaxpr.consts, *args
            )
            result = flat_results[0] if len(flat_results) == 1 else tuple(flat_results)
            return result, {}

        # Step 3: Evaluate the rewritten jaxpr.  The result is a flat
        # list: [original_outputs..., captured_sow_values...].
        all_flat_results = jax.core.eval_jaxpr(
            rewritten_jaxpr, closed_jaxpr.consts, *args
        )

        # Split into original outputs and captured intermediates.
        n_original_outputs = len(original_jaxpr.outvars)
        original_flat_results = all_flat_results[:n_original_outputs]
        captured_flat_values = all_flat_results[n_original_outputs:]

        # Step 4: Group captured flat values by (name, mode) and
        # reconstruct pytrees where applicable.
        intermediates: dict[str, dict[str, Any]] = {}
        offset = 0
        for (sow_name, sow_mode), n_leaves in zip(
            found_sow_keys, flat_counts, strict=True
        ):
            leaf_values = captured_flat_values[offset : offset + n_leaves]
            treedef = treedefs_snapshot.get(sow_name)
            if treedef is not None:
                reconstructed = jax.tree.unflatten(treedef, leaf_values)
            else:
                reconstructed = leaf_values[0] if n_leaves == 1 else tuple(leaf_values)
            intermediates.setdefault(sow_name, {})[sow_mode] = reconstructed
            offset += n_leaves

        result = (
            original_flat_results[0]
            if len(original_flat_results) == 1
            else tuple(original_flat_results)
        )
        return result, intermediates

    return wrapped
