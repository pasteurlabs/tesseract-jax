import warnings
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


def _merge_path(
    explicit_path: str, array_paths: Iterable[str]
) -> tuple[str, str | None]:
    """Merges and formats explicit path with array paths containing templates.

    Returns a tuple of (formatted_path, matched_template) where matched_template
    is the template string that matched, or None if no template matched.

    Examples:
        _merge_path('alpha.beta.x', ['alpha.beta.{}']) -> ('alpha.beta.{x}', 'alpha.beta.{}')
        _merge_path('delta.[2]', ['delta.[]']) -> ('delta.[2]', 'delta.[]')
        _merge_path('epsilon.k', ['alpha.{}']) -> ('epsilon.k', None)
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
            return ".".join(result_parts), array_path

    return explicit_path, None


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

        tesseract_path, matched_template = _merge_path(
            tesseract_path, schema_paths or []
        )

        flat_dict[tesseract_path] = val if matched_template else None

    return flat_dict


def _leaves_differ(returned: Any, expected: Any) -> bool:
    """Whether two static leaves disagree.

    Static leaves are decoded response data, so ``!=`` settles it for anything a
    served Tesseract can send: JSON has no type whose ``!=`` returns a non-bool.

    The fallback is for ``Tesseract.from_tesseract_api``, which hands back the
    objects the Python function built rather than a JSON round trip. An
    ``abstract_eval`` that reports a numpy array for a field is the case to
    picture: the field is not an aval, so it counts as static, and ``a != b``
    on two arrays is an array, which ``bool()`` refuses for anything but one
    element. Identity is then the only question left with an answer.
    """
    try:
        return bool(returned != expected)
    except (TypeError, ValueError):
        return returned is not expected


def warn_on_static_output_drift(
    paths: Sequence[Any],
    returned_values: Sequence[Any],
    expected_values: Sequence[Any],
) -> None:
    """Warn about static output leaves whose runtime value is not the traced one.

    ``apply`` runs after the trace, so a static leaf it returns arrives too late to
    be used: ``apply_tesseract`` hands back the value ``abstract_eval`` reported.
    A Tesseract that returns a different one from ``apply`` is therefore doing
    something the caller cannot observe, and silence would hide that.
    """
    for path, returned, expected in zip(
        paths, returned_values, expected_values, strict=True
    ):
        if not _leaves_differ(returned, expected):
            continue
        warnings.warn(
            f"Tesseract returned the static output {jax.tree_util.keystr(path)} as "
            f"{returned!r} from apply, but abstract_eval reported {expected!r}. "
            f"Static outputs are read at trace time, so the value from "
            f"abstract_eval is the one apply_tesseract returns and the value from "
            f"apply is ignored.",
            UserWarning,
            stacklevel=2,
        )


def dummy_output_tree(
    output_pytreedef: Any,
    n_avals: int,
    static_output_mask: Sequence[bool] = (),
) -> Any:
    """The output pytree carrying its own aval index at every array leaf.

    Static leaves get ``None``, which is an empty pytree node, so they vanish
    from anything that flattens this tree. That is what keeps the differentiable
    paths honest: ``_pytree_to_tesseract_flat`` never sees a static output, so
    every path-to-position map built from this tree still lines up with
    ``output_avals``, which holds arrays only.
    """
    if not any(static_output_mask):
        return jax.tree.unflatten(output_pytreedef, range(n_avals))
    idx = iter(range(n_avals))
    leaves = [None if static else next(idx) for static in static_output_mask]
    return jax.tree.unflatten(output_pytreedef, leaves)
