# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import jax.tree
from jax import ShapeDtypeStruct
from jax.tree_util import PyTreeDef
from jax.typing import ArrayLike
from tesseract_core import Tesseract

from tesseract_jax.tree_util import PyTree, _pytree_to_tesseract_flat, unflatten_args


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

        flat_tangents = _pytree_to_tesseract_flat(
            tangent_inputs, schema_paths=self.differentiable_input_paths
        )
        flat_tangents = {p: v for p, v in flat_tangents.items() if v is not None}

        output_flat = _pytree_to_tesseract_flat(
            jax.tree.unflatten(output_pytreedef, range(len(output_avals))),
            schema_paths=self.differentiable_output_paths,
        )

        jvp_outputs = [p for p, v in output_flat.items() if v is not None]

        out_data = self.client.jacobian_vector_product(
            inputs=primal_inputs,
            jvp_inputs=list(flat_tangents.keys()),
            jvp_outputs=jvp_outputs,
            tangent_vector=flat_tangents,
        )

        out = []
        for path, aval in zip(output_flat, output_avals, strict=False):
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

        flat_inputs = _pytree_to_tesseract_flat(
            primal_inputs, schema_paths=self.differentiable_input_paths
        )
        vjp_inputs = [
            p for p, m in zip(flat_inputs, is_static_mask, strict=True) if not m
        ]

        cotangent_pytree = jax.tree.unflatten(output_pytreedef, cotangents)
        flat_cotangents = _pytree_to_tesseract_flat(
            cotangent_pytree, schema_paths=self.differentiable_output_paths
        )
        cotangents_dict = {p: v for p, v in flat_cotangents.items() if v is not None}

        out_data = self.client.vector_jacobian_product(
            inputs=primal_inputs,
            vjp_inputs=vjp_inputs,
            vjp_outputs=list(cotangents_dict.keys()),
            cotangent_vector=cotangents_dict,
        )

        out_data = tuple(jax.tree.flatten(out_data)[0])
        return out_data
