from collections.abc import Iterable, Sequence
from typing import Any, TypeVar

import jax.tree
from jax.core import ShapedArray
from jax.tree_util import PyTreeDef
from jax.typing import ArrayLike

T = TypeVar("T")
type PyTree = Any


def split_args[T](
    flat_args: Sequence[T], mask: Sequence[bool]
) -> tuple[tuple[T, ...], tuple[T, ...]]:
    """Split a flat argument tuple according to mask (mask_False, mask_True)."""
    lists: tuple[list, list] = ([], [])
    for a, m in zip(flat_args, mask, strict=True):
        lists[m].append(a)
    return tuple(tuple(args) for args in lists)


def combine_args(args0: Sequence, args1: Sequence, mask: Sequence[bool]) -> tuple:
    """Merge the elements of two lists based on a mask.

    The length of the two lists combined is required to be equal to the length of the mask.
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
    array_args: tuple[ArrayLike | ShapedArray, ...],
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


def _split_path(path: str) -> list[str]:
    """Split a path on the dots that separate segments.

    A dot inside ``{...}`` belongs to the key, so ``a.{b.c}`` is two segments
    rather than three.
    """
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    for char in path:
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        if char == "." and depth == 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(char)
    parts.append("".join(buf))
    return parts


def _merge_path(
    explicit_path: str | Sequence[str], array_paths: Iterable[str]
) -> tuple[str, str | None]:
    """Merges and formats explicit path with array paths containing templates.

    Returns a tuple of (formatted_path, matched_template) where matched_template
    is the template string that matched, or None if no template matched.

    ``explicit_path`` may be given as the already-joined string or as the
    segments it was built from. A dict key is free to contain dots, so passing
    the segments is the only way to say where one ends; joining first and
    splitting again cannot tell ``{"a": {"b.c": v}}`` from ``{"a": {"b": {"c": v}}}``.

    Examples:
        _merge_path('alpha.beta.x', ['alpha.beta.{}']) -> ('alpha.beta.{x}', 'alpha.beta.{}')
        _merge_path('delta.[2]', ['delta.[]']) -> ('delta.[2]', 'delta.[]')
        _merge_path('epsilon.k', ['alpha.{}']) -> ('epsilon.k', None)
        _merge_path(['params', 'a.b'], ['params.{}']) -> ('params.{a.b}', 'params.{}')
    """
    if isinstance(explicit_path, str):
        explicit_parts = _split_path(explicit_path)
    else:
        explicit_parts = list(explicit_path)

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
                # Idempotent: batching re-merges paths this function produced.
                already = ep.startswith("{") and ep.endswith("}")
                result_parts.append(ep if already else f"{{{ep}}}")
            elif tp == "[]":
                result_parts.append(ep)  # already "[n]"
            else:
                matched = False
                break

        if matched:
            return ".".join(result_parts), array_path

    return ".".join(explicit_parts), None


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
        # Keep the segments rather than joining them: a dict key may itself
        # contain a dot, and joining first loses where the key ends.
        path_parts: list[str] = []
        for elem in jax_path:
            # for handling dicts
            if hasattr(elem, "key"):
                path_parts.append(str(elem.key))
            # for handling lists/tuples
            elif hasattr(elem, "idx"):
                path_parts.append(f"[{elem.idx}]")

        tesseract_path, matched_template = _merge_path(path_parts, schema_paths or [])

        flat_dict[tesseract_path] = val if matched_template else None

    return flat_dict
