from collections.abc import Sequence
from typing import Any, TypeAlias

import jax.tree
from jax.tree_util import PyTreeDef
from jax.typing import ArrayLike

PyTree: TypeAlias = Any


def combine_args(args0: Sequence, args1: Sequence, mask: Sequence[bool]) -> tuple:
    """Merge the elements of two lists based on a mask.

    The length of the two lists is required to be equal to the length of the mask.
    `combine_args` will populate the new list according to the mask: if the mask evaluates
    to `False` it will take the next item of the first list, if it evaluate to `True` it will
    take from the second list.

    Example:
        >>> combine_args(["foo", "bar"], [0, 1, 2], [1, 0, 0, 1, 1])
        [0, "foo", "bar", 1, 2]
    """
    assert sum(mask) == len(args1) and len(mask) - sum(mask) == len(args0)
    args0_iter, args1_iter = iter(args0), iter(args1)
    combined_args = [next(args1_iter) if m else next(args0_iter) for m in mask]
    return tuple(combined_args)


def unflatten_args(
    array_args: tuple[ArrayLike, ...],
    static_args: tuple[Any, ...],
    input_pytreedef: PyTreeDef,
    is_static_mask: tuple[bool, ...],
    remove_static_args: bool = False,
) -> PyTree:
    """Unflatten lists of arguments (static and not) into a pytree."""
    if remove_static_args:
        static_args_converted = [None] * len(static_args)
    else:
        static_args_converted = [
            elem.wrapped if hasattr(elem, "wrapped") else elem for elem in static_args
        ]

    combined_args = combine_args(array_args, static_args_converted, is_static_mask)
    result = jax.tree.unflatten(input_pytreedef, combined_args)

    if remove_static_args:
        result = _prune_nones(result)

    # Since jax 0.8, when tracing stuff without jit arrays are wrapped
    # by TypedNdArray (thin wrapper around a numpy array); this snippet converts them
    # back to ndarrays for downstream calculations.
    try:
        from jax._src.literals import TypedNdArray

        result = jax.tree.map(
            lambda v: v.val if isinstance(v, TypedNdArray) else v, result
        )

    except ImportError:
        pass

    return result


def _prune_nones(tree: PyTree) -> PyTree:
    if isinstance(tree, dict):
        return {k: _prune_nones(v) for k, v in tree.items() if v is not None}
    elif isinstance(tree, tuple | list):
        return type(tree)(_prune_nones(v) for v in tree if v is not None)
    else:
        return tree


def _merge_path(explicit_path: str, array_paths: list[str]) -> tuple[str, bool]:
    """Merges and formats explicit path with array paths containing templates.

    Returns a tuple of (formatted_path, matched) where matched indicates whether
    the path matched any template in array_paths.

    Examples:
        _merge_path('alpha.beta.x', ['alpha.beta.{}']) -> ('alpha.beta.{x}', True)
        _merge_path('delta.[2]', ['delta.[]']) -> ('delta.[2]', True)
        _merge_path('epsilon.k', ['alpha.{}']) -> ('epsilon.k', False)
    """
    explicit_parts = explicit_path.split(".")
    for array_path in array_paths:
        template_parts = array_path.split(".")
        if len(template_parts) != len(explicit_parts):
            continue

        result_parts = []
        matched = True
        for tp, ep in zip(template_parts, explicit_parts, strict=True):
            if tp == ep:
                result_parts.append(ep)
            elif tp == "{}":
                result_parts.append(f"{{{ep}}}")
            elif tp == "[]":
                result_parts.append(ep)  # already "[n]"
            else:
                matched = False
                break

        if matched:
            return ".".join(result_parts), True

    return explicit_path, False


def _pytree_to_tesseract_flat(
    pytree: PyTree, schema_paths: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Flatten a pytree to tesseract path format.

    Takes a pytree, flattens it and converts the flat paths
    into a Tesseract compatible format.
    For inputs that are differentiable, Tesseracts has the
    convention to wrap dict keys that are not pydantic models in curly braces {}.
    Furthermore, list indices are represented as .[index].

    Args:
        pytree: The pytree to flatten and convert.
        schema_paths: Optional dict from OpenAPI schema differentiable_arrays.
            Used to identify dict fields that need {key} formatting.

    Returns:
        Dict mapping tesseract path strings to values. Paths that don't match
        any template in schema_paths have their value set to None.
    """
    leaves = jax.tree_util.tree_flatten_with_path(pytree)[0]

    flat_dict = {}
    for jax_path, val in leaves:
        tesseract_path = ""
        for elem in jax_path:
            # for handling dicts
            if hasattr(elem, "key"):
                tesseract_path += f".{elem.key}"
            # for handling lists/tuples
            elif hasattr(elem, "idx"):
                tesseract_path += f".[{elem.idx}]"
        # remove leading dot
        tesseract_path = tesseract_path.lstrip(".")

        tesseract_path, is_differentiable = _merge_path(
            tesseract_path, list(schema_paths.keys()) if schema_paths else []
        )

        flat_dict[tesseract_path] = val if is_differentiable else None

    return flat_dict
