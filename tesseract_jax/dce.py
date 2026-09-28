# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dead-code-elimination rule for the ``tesseract_dispatch`` primitive.

AD requests the derivative of *every* differentiable leaf even when only a few
survive downstream (e.g. ``jacfwd`` of a function that returns one leaf of a
multi-output Tesseract, or ``jax.grad(...)[...]`` that keeps one input gradient).
JAX exposes the survivors to a primitive's DCE rule, letting us narrow the
requested sub-block (the ``live_*_paths`` for a ``jacobian``, ``live_output_paths``
for a ``jacobian_vector_product``, ``has_tangent`` for a
``vector_jacobian_product``) and drop the dead outvars so the Tesseract computes
only what is used.

Like :mod:`tesseract_jax.batching`, this module holds primitive-agnostic logic —
it operates purely on the ``JaxprEqn`` and never imports ``tesseract_dispatch_p``;
the rule is registered against the primitive in :mod:`tesseract_jax.primitive`.
"""

from collections.abc import Sequence
from typing import Any

from jax._src.interpreters import partial_eval as pe
from jax.tree_util import PyTreeDef

try:
    # ``DropVar`` lives in ``jax.core`` on our 0.7.0 lower bound and only moves to
    # ``jax.extend.core`` in later releases (importing it from ``jax.core`` warns
    # on 0.10+). ``JaxprEqn`` is in ``jax.extend.core`` across the whole range.
    from jax.extend.core import DropVar, JaxprEqn
except ImportError:  # pragma: no cover - exercised only on JAX 0.7.x
    from jax.core import DropVar
    from jax.extend.core import JaxprEqn

from tesseract_jax.tree_util import (
    dummy_output_tree,
    pytree_to_path_dict,
)


def live_jvp_output_positions(
    output_pytreedef: PyTreeDef,
    n_outputs: int,
    diff_output_paths: dict[str, Any],
    live_output_paths: tuple[str, ...] | None,
    static_output_mask: Sequence[bool] = (),
) -> list[int]:
    """Output-leaf positions a ``jacobian_vector_product`` bind should emit.

    Positions are returned in ``output_avals`` order so that abstract_eval, the
    endpoint wrapper and the DCE rule all agree on the layout. A leaf is kept
    when it is non-differentiable (its tangent is a cheap NaN and cannot be named
    by a path) or when its differentiable path is in ``live_output_paths``;
    ``live_output_paths is None`` means "keep everything".

    Static (non-array) output leaves never enter the bind, so ``static_output_mask``
    drops them from the layout via :func:`dummy_output_tree`; the positions returned
    then index ``output_avals``, which holds arrays only.
    """
    output_flat = pytree_to_path_dict(
        dummy_output_tree(output_pytreedef, n_outputs, static_output_mask),
        schema_paths=diff_output_paths,
    )
    positions = []
    for pos, (path, is_diff) in enumerate(output_flat.items()):
        if is_diff is None or live_output_paths is None or path in live_output_paths:
            positions.append(pos)
    return positions


def tesseract_dispatch_dce_rule(
    used_outputs: list[bool], eqn: JaxprEqn
) -> tuple[list[bool], JaxprEqn | None]:
    """Drop dead derivative outputs from a ``tesseract_dispatch`` equation.

    JAX surfaces which outputs survive downstream as ``used_outputs``; we narrow
    the requested sub-block (the ``live_*_paths`` for a ``jacobian``,
    ``live_output_paths`` for a ``jacobian_vector_product``, ``has_tangent`` for a
    ``vector_jacobian_product``) and drop the dead outvars so the Tesseract
    computes only what is used.

    ``apply`` carries no prunable output structure and defers to JAX's default
    rule. This optimization only kicks in when JAX runs DCE — i.e. under ``jit``
    (any mode) and un-jitted reverse mode; un-jitted ``jacfwd`` is unaffected.

    Equation invars are always kept: the endpoints evaluate the full primal
    regardless of which columns are differentiated, so pruning ``used_inputs``
    would be incorrect. A ``vector_jacobian_product`` prunes its outvars (one
    input-gradient per differentiated primal), never its invars.
    """
    # Effects-aware whole-equation drop (matches the un-pruned default exactly).
    if not any(used_outputs):
        return pe._default_dce_rule(used_outputs, eqn)

    eval_func = eqn.params["params"].eval_func
    if eval_func == "jacobian":
        return _dce_jacobian(used_outputs, eqn)
    if eval_func == "jacobian_vector_product":
        return _dce_jacobian_vector_product(used_outputs, eqn)
    if eval_func == "vector_jacobian_product":
        return _dce_vector_jacobian_product(used_outputs, eqn)
    return pe._default_dce_rule(used_outputs, eqn)


def _dce_jacobian(
    used_outputs: list[bool], eqn: JaxprEqn
) -> tuple[list[bool], JaxprEqn | None]:
    """Prune a ``jacobian`` equation's (out x in) block grid to its live rectangle."""
    dispatch_params = eqn.params["params"]
    in_paths = dispatch_params.live_input_paths
    out_paths = dispatch_params.live_output_paths
    if in_paths is None or out_paths is None:
        # No explicit path layout to map ``used_outputs`` onto; keep everything.
        return [True] * len(eqn.invars), eqn

    n_in, n_out = len(in_paths), len(out_paths)
    # Row-major layout: outvar ``i * n_in + j`` is block (out_path i, in_path j).
    used = [[used_outputs[i * n_in + j] for j in range(n_in)] for i in range(n_out)]
    live_out = [i for i in range(n_out) if any(used[i])]
    live_in = [j for j in range(n_in) if any(used[i][j] for i in range(n_out))]

    new_params = dict(
        eqn.params,
        params=dispatch_params.replace(
            live_output_paths=tuple(out_paths[i] for i in live_out),
            live_input_paths=tuple(in_paths[j] for j in live_in),
        ),
    )
    # Emit the live rectangle in the same row-major order abstract_eval expects.
    # Blocks inside the rectangle that are individually dead become DropVars.
    new_outvars = [
        eqn.outvars[i * n_in + j]
        if used_outputs[i * n_in + j]
        else DropVar(eqn.outvars[i * n_in + j].aval)
        for i in live_out
        for j in live_in
    ]
    return [True] * len(eqn.invars), eqn.replace(outvars=new_outvars, params=new_params)


def _dce_jacobian_vector_product(
    used_outputs: list[bool], eqn: JaxprEqn
) -> tuple[list[bool], JaxprEqn | None]:
    """Prune a ``jacobian_vector_product`` equation's dead output tangents."""
    dispatch_params = eqn.params["params"]
    client = dispatch_params.client
    output_pytreedef = dispatch_params.output_pytreedef
    n_outputs = len(dispatch_params.output_avals)
    static_output_mask = dispatch_params.static_output_mask
    diff_output_paths = client.differentiable_output_paths

    # Positions this bind currently emits (in output_avals order). Must line up
    # 1:1 with ``used_outputs`` / ``eqn.outvars``.
    cur_positions = live_jvp_output_positions(
        output_pytreedef,
        n_outputs,
        diff_output_paths,
        dispatch_params.live_output_paths,
        static_output_mask,
    )
    flat_items = list(
        pytree_to_path_dict(
            dummy_output_tree(output_pytreedef, n_outputs, static_output_mask),
            schema_paths=diff_output_paths,
        ).items()
    )

    new_outvars = []
    live_paths = []
    for k, pos in enumerate(cur_positions):
        path, is_diff = flat_items[pos]
        if is_diff is None:
            # Non-differentiable leaf: always retained (it has no path to name and
            # its tangent is a cheap NaN), but DropVar it when dead.
            ov = eqn.outvars[k]
            new_outvars.append(ov if used_outputs[k] else DropVar(ov.aval))
        elif used_outputs[k]:
            new_outvars.append(eqn.outvars[k])
            live_paths.append(path)
        # else: differentiable but dead -> dropped from the output contract.

    new_params = dict(
        eqn.params, params=dispatch_params.replace(live_output_paths=tuple(live_paths))
    )
    return [True] * len(eqn.invars), eqn.replace(outvars=new_outvars, params=new_params)


def _dce_vector_jacobian_product(
    used_outputs: list[bool], eqn: JaxprEqn
) -> tuple[list[bool], JaxprEqn | None]:
    """Prune a ``vector_jacobian_product`` equation's dead input gradients.

    A VJP bind returns one array per differentiated primal. A dead output means
    that primal's gradient is never consumed downstream, e.g.
    ``jax.grad(f)(inputs)["a"]``.

    ``has_tangent`` already exists to distinguish arrays that are closed over on
    the forward pass. Therefore, we can update this to also drop tangents
    recognised as dead by DCE after the reverse pass.
    """
    dispatch_params = eqn.params["params"]
    old_has_tangent = dispatch_params.has_tangent

    used_iter = iter(used_outputs)
    new_has_tangent = tuple(h and next(used_iter) for h in old_has_tangent)

    new_outvars = [ov for ov, u in zip(eqn.outvars, used_outputs, strict=True) if u]
    new_params = dict(
        eqn.params, params=dispatch_params.replace(has_tangent=new_has_tangent)
    )
    return [True] * len(eqn.invars), eqn.replace(outvars=new_outvars, params=new_params)
