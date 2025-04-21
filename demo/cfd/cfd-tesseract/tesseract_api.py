# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from functools import partial

import jax
import jax.numpy as jnp
import jax_cfd.base as cfd
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float32, ShapeDType

# TODO: !! Use JAX recipe for this, to avoid re-jitting of VJPs etc. !!


class InputSchema(BaseModel):
    v0: Differentiable[
        Array[
            (
                None,
                None,
                None,
            ),
            Float32,
        ]
    ] = Field(description="3D Array defining the initial velocity field [...]")
    density: float = Field(description="Density of the fluid")
    viscosity: float = Field(description="Viscosity of the fluid")
    inner_steps: float = Field(
        description="Number of solver steps for each timestep", default=25
    )
    outer_steps: float = Field(description="Number of timesteps steps", default=10)
    max_velocity: float = Field(description="Maximum velocity", default=2.0)
    cfl_safety_factor: float = Field(description="CFL safety factor", default=0.5)
    domain_size_x: float = Field(description="Domain size x", default=1.0)
    domain_size_y: float = Field(description="Domain size y", default=1.0)


class OutputSchema(BaseModel):
    result: Differentiable[
        Array[
            (
                None,
                None,
                None,
            ),
            Float32,
        ]
    ] = Field(description="3D Array defining the final velocity field [...]")


@partial(
    jax.jit,
    static_argnames=(
        "density",
        "viscosity",
        "inner_steps",
        "outer_steps",
        "max_velocity",
        "cfl_safety_factor",
        "domain_size_x",
        "domain_size_y",
    ),
)
def cfd_fwd(
    v0: jnp.ndarray,
    density: float,
    viscosity: float,
    inner_steps: int,
    outer_steps: int,
    max_velocity: float,
    cfl_safety_factor: float,
    domain_size_x: float,
    domain_size_y: float,
) -> tuple[jax.Array, jax.Array]:
    vx0 = v0[..., 0]
    vy0 = v0[..., 1]
    bc = cfd.boundaries.HomogeneousBoundaryConditions(
        (
            (cfd.boundaries.BCType.PERIODIC, cfd.boundaries.BCType.PERIODIC),
            (cfd.boundaries.BCType.PERIODIC, cfd.boundaries.BCType.PERIODIC),
        )
    )

    # reconstruct grid from input
    grid = cfd.grids.Grid(
        vx0.shape, domain=((0.0, domain_size_x), (0.0, domain_size_y))
    )

    vx0 = cfd.grids.GridArray(vx0, grid=grid, offset=(1.0, 0.5))
    vy0 = cfd.grids.GridArray(vy0, grid=grid, offset=(0.5, 1.0))

    # reconstrut GridVariable from input
    vx0 = cfd.grids.GridVariable(vx0, bc)
    vy0 = cfd.grids.GridVariable(vy0, bc)
    v0 = (vx0, vy0)

    # Choose a time step.
    dt = cfd.equations.stable_time_step(
        max_velocity, cfl_safety_factor, viscosity, grid
    )

    # Define a step function and use it to compute a trajectory.
    step_fn = cfd.funcutils.repeated(
        cfd.equations.semi_implicit_navier_stokes(
            density=density, viscosity=viscosity, dt=dt, grid=grid
        ),
        steps=inner_steps,
    )
    rollout_fn = jax.jit(cfd.funcutils.trajectory(step_fn, outer_steps))
    _, trajectory = jax.device_get(rollout_fn(v0))

    vxn = trajectory[0].array.data[-1]

    vyn = trajectory[1].array.data[-1]

    return jnp.stack([vxn, vyn], axis=-1)


def apply(inputs: InputSchema) -> OutputSchema:  #
    vn = cfd_fwd(
        v0=inputs.v0,
        density=inputs.density,
        viscosity=inputs.viscosity,
        inner_steps=inputs.inner_steps,
        outer_steps=inputs.outer_steps,
        max_velocity=inputs.max_velocity,
        cfl_safety_factor=inputs.cfl_safety_factor,
        domain_size_x=inputs.domain_size_x,
        domain_size_y=inputs.domain_size_y,
    )

    return OutputSchema(result=vn)


def abstract_eval(abstract_inputs):
    """Calculate output shape of apply from the shape of its inputs."""
    return {
        "result": ShapeDType(shape=abstract_inputs.v0.shape, dtype="float32"),
    }


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector,
):
    signature = [
        "v0",
        "density",
        "viscosity",
        "inner_steps",
        "outer_steps",
        "max_velocity",
        "cfl_safety_factor",
        "domain_size_x",
        "domain_size_y",
    ]
    # We need to do this, rather than just use jvp inputs, as the order in jvp_inputs
    # is not necessarily the same as the ordering of the args in the function signature.
    static_args = [arg for arg in signature if arg not in vjp_inputs]
    nonstatic_args = [arg for arg in signature if arg in vjp_inputs]

    def cfd_fwd_reordered(*args, **kwargs):
        return cfd_fwd(
            **{**{arg: args[i] for i, arg in enumerate(nonstatic_args)}, **kwargs}
        )

    out = {}
    if "result" in vjp_outputs:
        # Make the function depend only on nonstatic args, as jax.jvp
        # differentiates w.r.t. all free arguments.
        func = partial(
            cfd_fwd_reordered, **{arg: getattr(inputs, arg) for arg in static_args}
        )

        _, vjp_func = jax.vjp(
            func, *tuple(inputs.model_dump(include=vjp_inputs).values())
        )

        vals = vjp_func(cotangent_vector["result"])
        for arg, val in zip(nonstatic_args, vals, strict=False):
            out[arg] = out.get(arg, 0.0) + val

    return out
