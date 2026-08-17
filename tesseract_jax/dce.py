# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dead-code-elimination rule for the ``tesseract_dispatch`` primitive.

Forward-mode AD requests the derivative of *every* differentiable output even
when only a few survive downstream (e.g. ``jacfwd`` of a function that returns
one leaf of a multi-output Tesseract). JAX exposes the survivors to a primitive's
DCE rule, letting us narrow the requested sub-block (``live_output_paths`` /
``live_input_paths``) and drop the dead outvars so the Tesseract computes only
what is used.

Like :mod:`tesseract_jax.batching`, this module holds primitive-agnostic logic —
it operates purely on the ``JaxprEqn`` and never imports ``tesseract_dispatch_p``;
the rule is registered against the primitive in :mod:`tesseract_jax.primitive`.
"""

import jax.tree
from jax._src.interpreters import partial_eval as pe

try:
    # Preferred location (jax >= ~0.4.34, required on 0.10 to avoid a deprecation
    # warning). Falls back to ``jax.core`` on our 0.7.0 lower bound, where
    # ``jax.extend.core.DropVar`` does not yet exist.
    from jax.extend.core import DropVar, JaxprEqn
except ImportError:  # pragma: no cover - exercised only on older JAX
    from jax.core import DropVar
    from jax.extend.core import JaxprEqn

from tesseract_jax.tree_util import (
    _pytree_to_tesseract_flat,
    live_jvp_output_positions,
)


def tesseract_dispatch_dce_rule(
    used_outputs: list[bool], eqn: JaxprEqn
) -> tuple[list[bool], JaxprEqn | None]:
    """Drop dead derivative outputs from a ``tesseract_dispatch`` equation.

    JAX surfaces which outputs survive downstream as ``used_outputs``; we narrow
    the requested sub-block (via ``live_output_paths`` / ``live_input_paths``) and
    drop the dead outvars so the Tesseract computes only what is used.

    Only ``jacobian`` and ``jacobian_vector_product`` carry prunable output
    structure; ``apply`` and ``vector_jacobian_product`` defer to JAX's default
    rule. This optimisation only kicks in when JAX runs DCE — i.e. under ``jit``
    (any mode) and un-jitted reverse mode; un-jitted ``jacfwd`` is unaffected.

    Inputs are always kept: the endpoints evaluate the full primal regardless of
    which input columns are differentiated, so pruning ``used_inputs`` would be
    incorrect.
    """
    # Effects-aware whole-equation drop (matches the un-pruned default exactly).
    if not any(used_outputs):
        return pe._default_dce_rule(used_outputs, eqn)

    eval_func = eqn.params["eval_func"]
    if eval_func == "jacobian":
        return _dce_jacobian(used_outputs, eqn)
    if eval_func == "jacobian_vector_product":
        return _dce_jacobian_vector_product(used_outputs, eqn)
    return pe._default_dce_rule(used_outputs, eqn)


def _dce_jacobian(
    used_outputs: list[bool], eqn: JaxprEqn
) -> tuple[list[bool], JaxprEqn | None]:
    """Prune a ``jacobian`` equation's (out x in) block grid to its live rectangle."""
    in_paths = eqn.params.get("live_input_paths")
    out_paths = eqn.params.get("live_output_paths")
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
        live_output_paths=tuple(out_paths[i] for i in live_out),
        live_input_paths=tuple(in_paths[j] for j in live_in),
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
    params = eqn.params
    client = params["client"]
    output_pytreedef = params["output_pytreedef"]
    n_outputs = len(params["output_avals"])
    diff_output_paths = client.differentiable_output_paths

    # Positions this bind currently emits (in output_avals order). Must line up
    # 1:1 with ``used_outputs`` / ``eqn.outvars``.
    cur_positions = live_jvp_output_positions(
        output_pytreedef, n_outputs, diff_output_paths, params.get("live_output_paths")
    )
    flat_items = list(
        _pytree_to_tesseract_flat(
            jax.tree.unflatten(output_pytreedef, range(n_outputs)),
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

    new_params = dict(params, live_output_paths=tuple(live_paths))
    return [True] * len(eqn.invars), eqn.replace(outvars=new_outvars, params=new_params)
