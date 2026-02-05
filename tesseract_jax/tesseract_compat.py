# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from collections.abc import Sequence
from typing import Any, TypeAlias

import jax.tree
from jax import ShapeDtypeStruct
from jax.tree_util import PyTreeDef
from jax.typing import ArrayLike
from tesseract_core import Tesseract

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


def _merge_path(explicit_path: str, array_paths: list[str]):
    """Merges and formats explicit path with array paths containing templates.

    Examples:
        merge_path('alpha.beta.x', ['alpha.beta.{}']) -> 'alpha.beta.{x}'
        merge_path('delta[2]', ['delta.[]']) -> 'delta.[2]'
    """
    # Direct match
    if explicit_path in array_paths:
        return explicit_path

    for array_path in array_paths:
        # Replace template markers with regex patterns
        escaped_path = re.escape(array_path)
        pattern_str = escaped_path.replace(r"\{\}", r"(.+)").replace(
            r"\.\[\]", r"\[(.+)\]"
        )

        # Check if explicit path matches the pattern
        match = re.fullmatch(pattern_str, explicit_path)
        if match:
            # Extract the captured value
            captured_value = match.group(1)
            result = array_path.replace("{}", f"{{{captured_value}}}").replace(
                ".[]", f".[{captured_value}]"
            )
            return result

    return explicit_path


def _pytree_to_tesseract_flat(
    pytree: PyTree, schema_paths: dict[str, Any] | None = None
) -> list[tuple]:
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
        List of (path_string, value) tuples
    """
    leaves = jax.tree_util.tree_flatten_with_path(pytree)[0]

    flat_list = []
    for jax_path, val in leaves:
        # Extract path components
        flat_path = ""
        for elem in jax_path:
            # for handling dicts
            if hasattr(elem, "key"):
                flat_path += f".{elem.key}"
            # for handling lists/tuples
            elif hasattr(elem, "idx"):
                flat_path += f"[{elem.idx}]"
        # remove leading dot
        flat_path = flat_path.lstrip(".")

        flat_path = _merge_path(
            flat_path, list(schema_paths.keys()) if schema_paths else []
        )

        flat_list.append((flat_path, val))

    return flat_list


class Jaxeract:
    """A wrapper around a Tesseract client to make its signature compatible with JAX primitives."""

    def __init__(self, tesseract_client: Tesseract) -> None:
        """Initialize the Tesseract client."""
        self.client = tesseract_client

        self.tesseract_input_args = tuple(
            arg
            for arg in self.client.openapi_schema["components"]["schemas"][
                "Apply_InputSchema"
            ]["properties"]
        )
        # We need this to adhere to jax convention on tree flattening (sort keys alphabetically)
        # Only outermost level should be sufficient.
        self.tesseract_input_args = tuple(sorted(self.tesseract_input_args))

        self.tesseract_output_args = tuple(
            arg
            for arg in self.client.openapi_schema["components"]["schemas"][
                "Apply_OutputSchema"
            ]["properties"]
        )

        self.differentiable_input_paths = self.client.openapi_schema["components"][
            "schemas"
        ]["ApplyInputSchema"]["differentiable_arrays"]

        self.differentiable_output_paths = self.client.openapi_schema["components"][
            "schemas"
        ]["ApplyOutputSchema"]["differentiable_arrays"]

        self.available_methods = self.client.available_endpoints

    def abstract_eval(
        self,
        array_args: tuple[ArrayLike, ...],
        static_args: tuple[Any, ...],
        input_pytreedef: PyTreeDef,
        output_pytreedef: PyTreeDef | None,
        output_avals: tuple[ShapeDtypeStruct, ...] | None,
        is_static_mask: tuple[bool, ...],
    ) -> PyTree:
        """Run an abstract evaluation on a Tesseract.

        This used in order to get output shapes given input shapes.
        """
        avals = unflatten_args(array_args, static_args, input_pytreedef, is_static_mask)

        abstract_inputs = jax.tree.map(
            lambda x: (
                {"shape": x.shape, "dtype": x.dtype.name} if hasattr(x, "shape") else x
            ),
            avals,
        )

        out_data = self.client.abstract_eval(abstract_inputs)
        return out_data

    def apply(
        self,
        array_args: tuple[ArrayLike, ...],
        static_args: tuple[Any, ...],
        input_pytreedef: PyTreeDef,
        output_pytreedef: PyTreeDef | None,
        output_avals: tuple[ShapeDtypeStruct, ...] | None,
        is_static_mask: tuple[bool, ...],
    ) -> PyTree:
        """Call the Tesseract's apply endpoint with the given arguments."""
        inputs = unflatten_args(
            array_args, static_args, input_pytreedef, is_static_mask
        )

        out_data = self.client.apply(inputs)

        if output_avals is None:
            return out_data

        out_data = tuple(jax.tree.flatten(out_data)[0])
        return out_data

    def apply_pytree(
        self,
        inputs: PyTree,
    ) -> PyTree:
        """Call the Tesseract's apply endpoint with the given arguments."""
        return self.client.apply(inputs)

    def jacobian_vector_product(
        self,
        array_args: tuple[ArrayLike, ...],
        static_args: tuple[Any, ...],
        input_pytreedef: PyTreeDef,
        output_pytreedef: PyTreeDef,
        output_avals: tuple[ShapeDtypeStruct, ...],
        is_static_mask: tuple[bool, ...],
    ) -> PyTree:
        """Call the Tesseract's jvp endpoint with the given arguments."""
        n_primals = len(is_static_mask) - sum(is_static_mask)
        primals = array_args[:n_primals]
        tangents = array_args[n_primals:]

        primal_inputs = unflatten_args(
            primals, static_args, input_pytreedef, is_static_mask
        )
        tangent_inputs = unflatten_args(
            tangents,
            static_args,
            input_pytreedef,
            is_static_mask,
            remove_static_args=True,
        )

        flat_tangents = dict(
            _pytree_to_tesseract_flat(
                tangent_inputs, schema_paths=self.differentiable_input_paths
            )
        )

        jvp_inputs = list(flat_tangents.keys())
        jvp_outputs = list(self.differentiable_output_paths.keys())

        out_data = self.client.jacobian_vector_product(
            inputs=primal_inputs,
            jvp_inputs=jvp_inputs,
            jvp_outputs=jvp_outputs,
            tangent_vector=flat_tangents,
        )

        paths = [
            p
            for p, _ in _pytree_to_tesseract_flat(
                jax.tree.unflatten(output_pytreedef, range(len(output_avals))),
                schema_paths=self.differentiable_output_paths,
            )
        ]

        out = []
        for path, aval in zip(paths, output_avals, strict=False):
            if path in out_data:
                out.append(out_data[path])
            else:
                out.append(jax.numpy.full_like(aval, jax.numpy.nan))

        return tuple(out)

    def vector_jacobian_product(
        self,
        array_args: tuple[ArrayLike, ...],
        static_args: tuple[Any, ...],
        input_pytreedef: PyTreeDef,
        output_pytreedef: PyTreeDef,
        output_avals: tuple[ShapeDtypeStruct, ...],
        is_static_mask: tuple[bool, ...],
    ) -> PyTree:
        """Call the Tesseract's vjp endpoint with the given arguments."""
        n_primals = len(is_static_mask) - sum(is_static_mask)
        primals = array_args[:n_primals]
        cotangents = array_args[n_primals:]

        primal_inputs = unflatten_args(
            primals, static_args, input_pytreedef, is_static_mask
        )

        in_keys = [
            k
            for k, _ in _pytree_to_tesseract_flat(
                primal_inputs, schema_paths=self.differentiable_input_paths
            )
        ]
        vjp_inputs = [o for o, m in zip(in_keys, is_static_mask, strict=True) if not m]
        vjp_outputs = list(self.differentiable_output_paths.keys())

        paths = [
            p
            for p, _ in _pytree_to_tesseract_flat(
                jax.tree.unflatten(output_pytreedef, range(len(output_avals))),
                schema_paths=self.differentiable_output_paths,
            )
        ]

        cotangents_dict = {}

        for i, p in enumerate(paths):
            if p in vjp_outputs:
                cotangents_dict[p] = cotangents[i]

        out_data = self.client.vector_jacobian_product(
            inputs=primal_inputs,
            vjp_inputs=vjp_inputs,
            vjp_outputs=vjp_outputs,
            cotangent_vector=cotangents_dict,
        )

        out_data = tuple(jax.tree.flatten(out_data)[0])
        return out_data
