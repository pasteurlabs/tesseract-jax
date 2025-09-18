# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

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

    return result


def _prune_nones(tree: PyTree) -> PyTree:
    if isinstance(tree, dict):
        return {k: _prune_nones(v) for k, v in tree.items() if v is not None}
    elif isinstance(tree, tuple | list):
        return type(tree)(_prune_nones(v) for v in tree if v is not None)
    else:
        return tree


def _pytree_to_tesseract_flat(pytree: PyTree) -> list[tuple]:
    leaves = jax.tree_util.tree_flatten_with_path(pytree)[0]

    flat_list = []
    for jax_path, val in leaves:
        tesseract_path = ""
        first_elem = True
        for elem in jax_path:
            if hasattr(elem, "key"):
                if not first_elem:
                    tesseract_path += "."
                tesseract_path += elem.key
            elif hasattr(elem, "idx"):
                tesseract_path += f"[{elem.idx}]"
            first_elem = False
        flat_list.append((tesseract_path, val))

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
        output_pytreedef: PyTreeDef,
        output_avals: tuple[ShapeDtypeStruct, ...],
        is_static_mask: tuple[bool, ...],
    ) -> PyTree:
        """Call the Tesseract's apply endpoint with the given arguments."""
        inputs = unflatten_args(
            array_args, static_args, input_pytreedef, is_static_mask
        )

        out_data = self.client.apply(inputs)

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

        flat_tangents = dict(_pytree_to_tesseract_flat(tangent_inputs))

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
                jax.tree.unflatten(output_pytreedef, range(len(output_avals)))
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

        in_keys = [k for k, _ in _pytree_to_tesseract_flat(primal_inputs)]
        vjp_inputs = [o for o, m in zip(in_keys, is_static_mask, strict=True) if not m]
        vjp_outputs = list(self.differentiable_output_paths.keys())

        paths = [
            p
            for p, _ in _pytree_to_tesseract_flat(
                jax.tree.unflatten(output_pytreedef, range(len(output_avals)))
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
