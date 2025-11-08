from typing import Any

import numpy as np
import trimesh
from pydantic import BaseModel, Field
from pysdf import SDF
from tesseract_core.runtime import Array, Differentiable, Float32, Int32, ShapeDType
from tesseract_core.runtime.experimental import TesseractReference

#
# Schemata
#


class InputSchema(BaseModel):
    """Input schema for bar geometry design and SDF generation."""

    differentiable_parameters: Differentiable[
        Array[
            (None,),
            Float32,
        ]
    ] = Field(
        description="Flattened array of geometry parameters that are passed to the mesh_tesseract."
    )

    non_differentiable_parameters: Array[
        (None,),
        Float32,
    ] = Field(description="Flattened array of non-differentiable geometry parameters.")

    static_parameters: list[int] = Field(
        description=("List of static integers used to construct the geometry.")
    )

    string_parameters: list[str] = Field(
        description=("List of string parameters used to construct the geometry.")
    )

    mesh_tesseract: TesseractReference = Field(description="Tesseract to call.")

    scale_mesh: float = Field(
        default=1.0,
        description="Scaling factor applied to the generated mesh.",
    )

    grid_size: list[float] = Field(
        description="Size of the bounding box in x, y, z directions."
    )

    grid_elements: list[int] = Field(
        description="Number of elements in the bounding box in x, y, z directions."
    )

    grid_center: list[float] = Field(
        description="Center of the bounding box in x, y, z directions."
    )

    epsilon: float = Field(
        default=1e-5,
        description=(
            "Epsilon value for finite difference approximation of the Jacobian. "
        ),
    )
    normalize_jacobian: bool = Field(
        default=False,
        description=("Whether to normalize the Jacobian by the number of elements"),
    )


class TriangularMesh(BaseModel):
    """Triangular mesh representation with fixed-size arrays."""

    points: Array[(None, 3), Float32] = Field(description="Array of vertex positions.")
    faces: Array[(None, 3), Float32] = Field(
        description="Array of triangular faces defined by indices into the points array."
    )
    n_points: Int32 = Field(
        default=0, description="Number of valid points in the points array."
    )
    n_faces: Int32 = Field(
        default=0, description="Number of valid faces in the faces array."
    )


class OutputSchema(BaseModel):
    """Output schema for generated geometry and SDF field."""

    mesh: TriangularMesh = Field(
        description="Triangular mesh representation of the geometry"
    )
    sdf: Differentiable[
        Array[
            (None, None, None),
            Float32,
        ]
    ] = Field(description="SDF field of the geometry")


#
# Helper functions
#


def get_geometry(
    target: TesseractReference,
    differentiable_parameters: np.ndarray,
    non_differentiable_parameters: np.ndarray,
    static_parameters: list[int],
    string_parameters: list[str],
) -> trimesh.Trimesh:
    """Build a pyvista geometry from the parameters.

    The parameters are expected to be of shape (n_chains, n_edges_per_chain + 1, 3),
    """
    mesh = target.apply(
        {
            "differentiable_parameters": differentiable_parameters,
            "non_differentiable_parameters": non_differentiable_parameters,
            "static_parameters": static_parameters,
            "string_parameters": string_parameters,
        }
    )["mesh"]

    mesh = trimesh.Trimesh(
        vertices=mesh["points"],
        faces=mesh["faces"],
    )

    return mesh


def compute_sdf(
    geometry: trimesh.Trimesh,
    grid_center: list[float],
    Lx: float,
    Ly: float,
    Lz: float,
    Nx: int,
    Ny: int,
    Nz: int,
) -> np.ndarray:
    """Create a pyvista plane that has the SDF values stored as a vertex attribute.

    The SDF field is computed based on the geometry defined by the parameters.
    """
    x, y, z = np.meshgrid(
        np.linspace(-Lx / 2, Lx / 2, Nx) + grid_center[0],
        np.linspace(-Ly / 2, Ly / 2, Ny) + grid_center[1],
        np.linspace(-Lz / 2, Lz / 2, Nz) + grid_center[2],
        indexing="ij",
    )

    points, faces = geometry.vertices, geometry.faces

    sdf_function = SDF(points, faces)

    grid_points = np.vstack((x.ravel(), y.ravel(), z.ravel())).T
    sdf_values = sdf_function(grid_points).astype(np.float32)

    sd_field = sdf_values.reshape((Nx, Ny, Nz))

    return -sd_field


def apply_fn(
    target: TesseractReference,
    differentiable_parameters: np.ndarray,
    non_differentiable_parameters: np.ndarray,
    static_parameters: list[int],
    string_parameters: list[str],
    scale_mesh: float,
    grid_size: np.ndarray,
    grid_elements: np.ndarray,
    grid_center: np.ndarray,
) -> tuple[np.ndarray, trimesh.Trimesh]:
    """Get the sdf values of a the geometry defined by the parameters as a 2D array."""
    geo = get_geometry(
        target=target,
        differentiable_parameters=differentiable_parameters,
        non_differentiable_parameters=non_differentiable_parameters,
        static_parameters=static_parameters,
        string_parameters=string_parameters,
    )
    # scale the mesh
    geo = geo.apply_scale(scale_mesh)

    sd_field = compute_sdf(
        geo,
        grid_center=grid_center,
        Lx=grid_size[0],
        Ly=grid_size[1],
        Lz=grid_size[2],
        Nx=grid_elements[0],
        Ny=grid_elements[1],
        Nz=grid_elements[2],
    )

    return sd_field, geo


def jac_sdf_wrt_params(
    target: TesseractReference,
    differentiable_parameters: np.ndarray,
    non_differentiable_parameters: np.ndarray,
    static_parameters: list[int],
    string_parameters: list[str],
    scale_mesh: float,
    grid_size: np.ndarray,
    grid_elements: np.ndarray,
    grid_center: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    """Compute the Jacobian of the SDF values with respect to the parameters.

    The Jacobian is computed by finite differences.
    The shape of the Jacobian is (n_chains, n_edges_per_chain + 1, 3, Nx, Ny).
    """
    jac = np.zeros(
        (
            differentiable_parameters.size,
            grid_elements[0],
            grid_elements[1],
            grid_elements[2],
        )
    )

    sd_field_base, _ = apply_fn(
        target=target,
        differentiable_parameters=differentiable_parameters,
        non_differentiable_parameters=non_differentiable_parameters,
        static_parameters=static_parameters,
        string_parameters=string_parameters,
        scale_mesh=scale_mesh,
        grid_elements=grid_elements,
        grid_size=grid_size,
        grid_center=grid_center,
    )

    for i in range(differentiable_parameters.size):
        # we only care about the y and z coordinate

        params_eps = differentiable_parameters.copy()
        params_eps[i] += epsilon

        sdf_epsilon, _ = apply_fn(
            target=target,
            differentiable_parameters=params_eps,
            non_differentiable_parameters=non_differentiable_parameters,
            static_parameters=static_parameters,
            string_parameters=string_parameters,
            scale_mesh=scale_mesh,
            grid_elements=grid_elements,
            grid_size=grid_size,
            grid_center=grid_center,
        )
        jac[i] = (sdf_epsilon - sd_field_base) / epsilon

    return jac


#
# Tesseract endpoints
#

N_POINTS = 1000
N_FACES = 2000


def apply(inputs: InputSchema) -> OutputSchema:
    """Generate mesh and SDF from bar geometry parameters.

    Args:
        inputs: Input schema containing bar geometry parameters.

    Returns:
        Output schema with generated mesh and SDF field.
    """
    sdf, mesh = apply_fn(
        target=inputs.mesh_tesseract,
        differentiable_parameters=inputs.differentiable_parameters,
        non_differentiable_parameters=inputs.non_differentiable_parameters,
        grid_size=inputs.grid_size,
        static_parameters=inputs.static_parameters,
        string_parameters=inputs.string_parameters,
        scale_mesh=inputs.scale_mesh,
        grid_elements=inputs.grid_elements,
        grid_center=inputs.grid_center,
    )

    points = np.zeros((N_POINTS, 3), dtype=np.float32)
    faces = np.zeros((N_FACES, 3), dtype=np.float32)

    points[: mesh.vertices.shape[0], :] = mesh.vertices.astype(np.float32)
    faces[: mesh.faces.shape[0], :] = mesh.faces.astype(np.int32)

    return OutputSchema(
        sdf=sdf,
        mesh=TriangularMesh(
            points=points,
            faces=faces,
            n_points=mesh.vertices.shape[0],
            n_faces=mesh.faces.shape[0],
        ),
    )


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
) -> dict[str, Any]:
    """Compute vector-Jacobian product for backpropagation.

    Args:
        inputs: Input schema containing bar geometry parameters.
        vjp_inputs: Set of input variable names for gradient computation.
        vjp_outputs: Set of output variable names for gradient computation.
        cotangent_vector: Cotangent vectors for the specified outputs.

    Returns:
        Dictionary containing VJP for the specified inputs.
    """
    assert vjp_inputs == {"differentiable_parameters"}
    assert vjp_outputs == {"sdf"}

    jac = jac_sdf_wrt_params(
        target=inputs.mesh_tesseract,
        differentiable_parameters=inputs.differentiable_parameters,
        non_differentiable_parameters=inputs.non_differentiable_parameters,
        static_parameters=inputs.static_parameters,
        string_parameters=inputs.string_parameters,
        scale_mesh=inputs.scale_mesh,
        grid_size=inputs.grid_size,
        grid_elements=inputs.grid_elements,
        epsilon=inputs.epsilon,
        grid_center=inputs.grid_center,
    )
    if inputs.normalize_jacobian:
        n_elements = (
            inputs.grid_elements[0] * inputs.grid_elements[1] * inputs.grid_elements[2]
        )
        jac = jac / n_elements
    # Reduce the cotangent vector to the shape of the Jacobian, to compute VJP by hand
    vjp = np.einsum("klmn,lmn->k", jac, cotangent_vector["sdf"]).astype(np.float32)

    return {"differentiable_parameters": vjp}


def abstract_eval(abstract_inputs: InputSchema) -> dict:
    """Calculate output shape of apply from the shape of its inputs.

    Args:
        abstract_inputs: Input schema with parameter shapes.

    Returns:
        Dictionary describing output shapes and dtypes.
    """
    return {
        "sdf": ShapeDType(
            shape=(
                abstract_inputs.grid_elements[0],
                abstract_inputs.grid_elements[1],
                abstract_inputs.grid_elements[2],
            ),
            dtype="float32",
        ),
        "mesh": {
            "points": ShapeDType(shape=(N_POINTS, 3), dtype="float32"),
            "faces": ShapeDType(shape=(N_FACES, 3), dtype="float32"),
            "n_points": ShapeDType(shape=(), dtype="int32"),
            "n_faces": ShapeDType(shape=(), dtype="int32"),
        },
    }
