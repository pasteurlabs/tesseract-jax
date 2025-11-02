import os
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import meshio
from jax_fem.generate_mesh import Mesh, box_mesh, get_meshio_cell_type

# Import JAX-FEM specific modules
from jax_fem.problem import Problem
from jax_fem.solver import ad_wrapper
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float32, Int32, ShapeDType
from tesseract_core.runtime.tree_transforms import filter_func, flatten_with_paths

crt_file_path = os.path.dirname(__file__)
data_dir = os.path.join(crt_file_path, "data")
#
# Schemata
#


class HexMesh(BaseModel):
    points: Array[(None, 3), Float32] = Field(description="Array of vertex positions.")
    faces: Array[(None, 8), Int32] = Field(
        description="Array of hexahedral faces defined by indices into the points array."
    )
    n_points: Int32 = Field(
        default=0, description="Number of valid points in the points array."
    )
    n_faces: Int32 = Field(
        default=0, description="Number of valid faces in the faces array."
    )


class InputSchema(BaseModel):
    rho: Differentiable[
        Array[
            (
                None,
                None,
            ),
            Float32,
        ]
    ] = Field(description="2D density field for topology optimization")
    Lx: float = Field(
        default=60.0, description="Length of the simulation box in the x direction."
    )
    Ly: float = Field(
        default=30.0,
        description=("Length of the simulation box in the y direction."),
    )
    Lz: float = Field(
        default=30.0, description="Length of the simulation box in the z direction."
    )
    Nx: int = Field(
        default=60,
        description=("Number of elements in the x direction."),
    )
    Ny: int = Field(
        default=30,
        description=("Number of elements in the y direction."),
    )
    Nz: int = Field(
        default=30,
        description=("Number of elements in the z direction."),
    )
    hex_mesh: HexMesh = Field(
        description="Hexahedral mesh representation of the geometry",
    )
    use_regular_grid: bool = Field(
        description="Toggle to use a regular grid mesh instead of imported mesh",
    )
    van_neumann_mask: Array[(None,), Int32] = Field(
        description="Mask for van Neumann boundary conditions",
    )
    van_neumann_values: Array[(None, None), Float32] = Field(
        description="Values for van Neumann boundary conditions",
    )
    dirichlet_mask: Array[(None,), Int32] = Field(
        description="Mask for Dirichlet boundary conditions",
    )
    dirichlet_values: Array[(None,), Float32] = Field(
        description="Values for Dirichlet boundary conditions",
    )


class OutputSchema(BaseModel):
    compliance: Differentiable[
        Array[
            (),
            Float32,
        ]
    ] = Field(description="Compliance of the structure, a measure of stiffness")


#
# Helper functions
#


# Define constitutive relationship
# Adapted from JAX-FEM
# https://github.com/deepmodeling/jax-fem/blob/1bdbf060bb32951d04ed9848c238c9a470fee1b4/demos/topology_optimization/example.py
class Elasticity(Problem):
    def custom_init(self, van_neumann_value_fns: list[Callable]):
        self.fe = self.fes[0]
        self.fe.flex_inds = jnp.arange(len(self.fe.cells))

        self.van_neumann_value_fns = van_neumann_value_fns


    def get_tensor_map(self):
        def stress(u_grad, theta):
            Emax = 70.0e3
            Emin = 1e-3 * Emax
            nu = 0.3
            penal = 3.0
            E = Emin + (Emax - Emin) * theta[0] ** penal
            epsilon = 0.5 * (u_grad + u_grad.T)
            # eps11 = epsilon[0, 0]
            # eps22 = epsilon[1, 1]
            # eps12 = epsilon[0, 1]
            # mu = E / (2 * (1 + nu))
            # sigma = jnp.trace(epsilon) * jnp.eye(self.dim) + 2*mu*epsilon
            # # sig11 = E / (1 + nu) / (1 - nu) * (eps11 + nu * eps22)
            # # sig22 = E / (1 + nu) / (1 - nu) * (nu * eps11 + eps22)
            # # sig12 = E / (1 + nu) * eps12
            # # sigma = jnp.array([[sig11, sig12], [sig12, sig22]])

            # Correct 3D linear elasticity constitutive law
            # Lamé parameters
            lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))  # First Lamé parameter
            mu = E / (2.0 * (1.0 + nu))  # Second Lamé parameter (shear modulus)

            # Stress-strain relationship
            sigma = lmbda * jnp.trace(epsilon) * jnp.eye(self.dim) + 2.0 * mu * epsilon

            return sigma

        return stress

    def get_surface_maps(self):
        # def surface_map(u, x):
        #     return jnp.array([0.0, 0.0, 100.0])

        # return [surface_map]

        return self.van_neumann_value_fns

    def set_params(self, params):
        # Override base class method.
        full_params = jnp.ones((self.fe.num_cells, params.shape[1]))
        full_params = full_params.at[self.fe.flex_inds].set(params)
        thetas = jnp.repeat(full_params[:, None, :], self.fe.num_quads, axis=1)
        self.full_params = full_params
        self.internal_vars = [thetas]

    def compute_compliance(self, sol):
        # Surface integral
        boundary_inds = self.boundary_inds_list[0]
        _, nanson_scale = self.fe.get_face_shape_grads(boundary_inds)
        u_face = (
            sol[self.fe.cells][boundary_inds[:, 0]][:, None, :, :]
            * self.fe.face_shape_vals[boundary_inds[:, 1]][:, :, :, None]
        )
        u_face = jnp.sum(u_face, axis=2)
        subset_quad_points = self.physical_surface_quad_points[0]
        neumann_fn = self.get_surface_maps()[0]
        traction = jax.vmap(jax.vmap(neumann_fn))(u_face, subset_quad_points)
        val = jnp.sum(traction * u_face * nanson_scale[:, :, None])
        return val


# Memoize the setup function to avoid expensive recomputation
# @lru_cache(maxsize=1)
def setup(
    Nx: int = 60,
    Ny: int = 30,
    Nz: int = 30,
    Lx: float = 60.0,
    Ly: float = 30.0,
    Lz: float = 30.0,
    pts: jnp.ndarray = None,
    cells: jnp.ndarray = None,
    dirichlet_mask: jnp.ndarray = None,
    dirichlet_values: jnp.ndarray = None,
    van_neumann_mask: jnp.ndarray = None,
    van_neumann_values: jnp.ndarray = None,
) -> tuple[Elasticity, Callable]:
    # Specify mesh-related information. We use a structured box mesh here.
    ele_type = "HEX8"
    if pts is None and cells is None:
        cell_type = get_meshio_cell_type(ele_type)
        meshio_mesh = box_mesh(
            Nx=Nx, Ny=Ny, Nz=Nz, domain_x=Lx, domain_y=Ly, domain_z=Lz
        )
        mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])
    else:
        meshio_mesh = meshio.Mesh(points=pts, cells={'hexahedron': cells})
        mesh = Mesh(pts, meshio_mesh.cells_dict['hexahedron'])

    def bc_factory(
        masks: jnp.ndarray,
        values: jnp.ndarray,
        is_van_neumann: bool = False,
    ) -> tuple[list[Callable], list[Callable]]:
        location_functions = []
        value_functions = []
        for i in range(values.shape[0]):

            def location_fn(point, index):
                # return mask[index]
                return jax.lax.dynamic_index_in_dim(masks, index, 0, keepdims=False) == i
            
            def value_fn(point):
                return values[i]
            
            def value_fn_vn(u, x):
                return values[i]
            
            location_functions.append(location_fn)
            value_functions.append(value_fn_vn if is_van_neumann else value_fn)

        return location_functions, value_functions

    dirichlet_location_fns, dirichlet_value_fns = bc_factory(
        dirichlet_mask, dirichlet_values
    )

    van_neumann_locations, van_neumann_value_fns = bc_factory(
        van_neumann_mask, van_neumann_values, is_van_neumann=True
    )

    dirichlet_bc_info = [dirichlet_location_fns * 3, [0, 1, 2], dirichlet_value_fns * 3]

    location_fns = van_neumann_locations

    # Define boundary conditions and values.
    # def fixed_location(point):
    #     return jnp.isclose(point[0], 0, atol=1e-5)

    # print(Lx, Ly, Lz)
    # print(f"Mesh pts bounds: x[{mesh.points[:,0].min()},{mesh.points[:,0].max()}], y[{mesh.points[:,1].min()},{mesh.points[:,1].max()}], z[{mesh.points[:,2].min()},{mesh.points[:,2].max()}]")
    # def fixed_location(point):
    #     # return jnp.isclose(point[0], -Lx / 3, atol=0.1)
    #     return point[0] < (-Lx / 2 + 1e-5)  # Left face

    # def load_location(point):

    #     # return jnp.logical_and(
    #     #     jnp.logical_and(
    #     #         jnp.isclose(point[0], Lx / 2, atol=1e-5),
    #     #         jnp.isclose(point[1], -Ly / 2, atol=1e-5),
    #     #     ),
    #     #     jnp.isclose(point[2], Lz / 2, atol=1e-5),
    #     # )
        
    #     return jnp.logical_and(
    #         jnp.isclose(point[0], 0, atol=1e-5),
    #         jnp.isclose(point[1], 0, atol=0.1 * Ly + 1e-5),
    #     )

    # def dirichlet_val(point):
    #     return 0.0

            
    # # # Define boundary conditions and values.
    # def fixed_location(point, index):
    #     return jnp.isclose(point[0], -Lx/2, atol=0.1)

    # def load_location(point):
    #     return jnp.logical_and(jnp.logical_and(
    #         jnp.isclose(point[0], Lx/2, atol=1e-2),
    #         jnp.isclose(point[2], -Lz/2, atol=1e-2),
    #     ), jnp.isclose(point[1], Ly/2, atol=1e-2))

    # def dirichlet_val(point):
    #     return 0.0

    # dirichlet_bc_info = [[fixed_location] * 3, [0, 1, 2], [dirichlet_val] * 3]

    # location_fns = [load_location]

    # Define forward problem
    problem = Elasticity(
        mesh,
        vec=3,
        dim=3,
        ele_type=ele_type,
        dirichlet_bc_info=dirichlet_bc_info,
        location_fns=location_fns,
        additional_info=(van_neumann_value_fns,)
        # additional_info=([0.1],)
    )

    # Apply the automatic differentiation wrapper
    # This is a critical step that makes the problem solver differentiable
    fwd_pred = ad_wrapper(
        problem,
        solver_options={"umfpack_solver": {}},
        adjoint_solver_options={"umfpack_solver": {}},
    )
    return problem, fwd_pred


def apply_fn(inputs: dict) -> dict:
    """Compute the compliance of the structure given a density field."""
    if not inputs["use_regular_grid"]:
        problem, fwd_pred = setup(
            Nx=inputs["Nx"],
            Ny=inputs["Ny"],
            Nz=inputs["Nz"],
            Lx=inputs["Lx"],
            Ly=inputs["Ly"],
            Lz=inputs["Lz"],
            pts=inputs["hex_mesh"]["points"][: inputs["hex_mesh"]["n_points"]],
            cells=inputs["hex_mesh"]["faces"][: inputs["hex_mesh"]["n_faces"]],
            dirichlet_mask=inputs["dirichlet_mask"],
            dirichlet_values=inputs["dirichlet_values"],
            van_neumann_mask=inputs["van_neumann_mask"],
            van_neumann_values=inputs["van_neumann_values"],
        )
    else:
        problem, fwd_pred = setup(
            Nx=inputs["Nx"],
            Ny=inputs["Ny"],
            Nz=inputs["Nz"],
            Lx=inputs["Lx"],
            Ly=inputs["Ly"],
            Lz=inputs["Lz"],
        )
    print(f"Setup completed with mesh of {problem.fe.num_cells} elements.")
    if inputs["use_regular_grid"]:
        rho = inputs["rho"]
    else:
        rho = inputs["rho"][: inputs["hex_mesh"]["n_faces"]]
    # print(rho)

    sol_list = fwd_pred(rho)
    compliance = problem.compute_compliance(sol_list[0])
    return {"compliance": compliance.astype(jnp.float32)}


#
# Tesseract endpoints
#


def apply(inputs: InputSchema) -> OutputSchema:
    return apply_fn(inputs.model_dump())


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
):
    assert vjp_inputs == {"rho"}
    assert vjp_outputs == {"compliance"}

    inputs = inputs.model_dump()

    filtered_apply = filter_func(apply_fn, inputs, vjp_outputs)
    _, vjp_func = jax.vjp(
        filtered_apply, flatten_with_paths(inputs, include_paths=vjp_inputs)
    )
    out = vjp_func(cotangent_vector)[0]
    return out


def abstract_eval(abstract_inputs):
    """Calculate output shape of apply from the shape of its inputs."""
    return {"compliance": ShapeDType(shape=(), dtype="float32")}
