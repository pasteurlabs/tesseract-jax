from functools import lru_cache
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
    max_points: int = Field(
        default=1000,
        description=("Maximum number of points in the output mesh."),
    )
    max_faces: int = Field(
        default=2000,
        description=("Maximum number of faces in the output mesh."),
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


def get_geometries(
    target: TesseractReference,
    differentiable_parameters: list[np.ndarray],
    non_differentiable_parameters: list[np.ndarray],
    static_parameters: list[list[int]],
    string_parameters: list[str],
) -> list[trimesh.Trimesh]:
    """Call the tesseract reference to get the geometries."""
    meshes = target.apply(
        {
            "differentiable_parameters": differentiable_parameters,
            "non_differentiable_parameters": non_differentiable_parameters,
            "static_parameters": static_parameters,
            "string_parameters": string_parameters,
        }
    )["meshes"]

    meshes = [
        trimesh.Trimesh(
            vertices=mesh["points"],
            faces=mesh["faces"],
        )
        for mesh in meshes
    ]

    return meshes


@lru_cache(maxsize=1)
def grid_points(
    Lx: float,
    Ly: float,
    Lz: float,
    Nx: int,
    Ny: int,
    Nz: int,
    grid_center: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a regular grid in 3D space."""
    x, y, z = np.meshgrid(
        np.linspace(-Lx / 2, Lx / 2, Nx) + grid_center[0],
        np.linspace(-Ly / 2, Ly / 2, Ny) + grid_center[1],
        np.linspace(-Lz / 2, Lz / 2, Nz) + grid_center[2],
        indexing="ij",
    )

    return np.vstack((x.ravel(), y.ravel(), z.ravel())).T


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
    points, faces = geometry.vertices, geometry.faces

    sdf_function = SDF(points, faces)

    points = grid_points(
        Lx=Lx,
        Ly=Ly,
        Lz=Lz,
        Nx=Nx,
        Ny=Ny,
        Nz=Nz,
        grid_center=tuple(grid_center),
    )

    sdf_values = sdf_function(points).astype(np.float32)

    sd_field = sdf_values.reshape((Nx, Ny, Nz))

    return -sd_field


def geometries_and_sdf(
    target: TesseractReference,
    differentiable_parameters: list[np.ndarray],
    non_differentiable_parameters: list[np.ndarray],
    static_parameters: list[list[int]],
    string_parameters: list[str],
    scale_mesh: float,
    grid_size: np.ndarray,
    grid_elements: np.ndarray,
    grid_center: np.ndarray,
) -> tuple[list[np.ndarray], list[trimesh.Trimesh]]:
    """Get the sdf values of a the geometry defined by the parameters as a 2D array."""
    geos = get_geometries(
        target=target,
        differentiable_parameters=differentiable_parameters,
        non_differentiable_parameters=non_differentiable_parameters,
        static_parameters=static_parameters,
        string_parameters=string_parameters,
    )
    # scale the mesh
    geos = [geo.apply_scale(scale_mesh) for geo in geos]

    sd_fields = [
        compute_sdf(
            geo,
            grid_center=grid_center,
            Lx=grid_size[0],
            Ly=grid_size[1],
            Lz=grid_size[2],
            Nx=grid_elements[0],
            Ny=grid_elements[1],
            Nz=grid_elements[2],
        )
        for geo in geos
    ]

    return sd_fields, geos


#
# Tesseract endpoints
#


def apply(inputs: InputSchema) -> OutputSchema:
    """Generate mesh and SDF from bar geometry parameters.

    Args:
        inputs: Input schema containing bar geometry parameters.

    Returns:
        Output schema with generated mesh and SDF field.
    """
    sdf, mesh = geometries_and_sdf(
        target=inputs.mesh_tesseract,
        differentiable_parameters=[inputs.differentiable_parameters],
        non_differentiable_parameters=[inputs.non_differentiable_parameters],
        static_parameters=[inputs.static_parameters],
        string_parameters=inputs.string_parameters,
        grid_size=inputs.grid_size,
        scale_mesh=inputs.scale_mesh,
        grid_elements=inputs.grid_elements,
        grid_center=inputs.grid_center,
    )

    sdf = sdf[0]  # only one geometry
    mesh = mesh[0]

    points = np.zeros((inputs.max_points, 3), dtype=np.float32)
    faces = np.zeros((inputs.max_faces, 3), dtype=np.float32)

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

    params = []
    params.append(differentiable_parameters.copy())

    n_params = differentiable_parameters.size

    for i in range(n_params):
        # we only care about the y and z coordinate
        params_eps = differentiable_parameters.copy()
        params_eps[i] += epsilon
        params.append(params_eps)

    sdf_fields, _ = geometries_and_sdf(
        target=target,
        differentiable_parameters=params,
        non_differentiable_parameters=[non_differentiable_parameters] * (n_params + 1),
        static_parameters=[static_parameters] * (n_params + 1),
        string_parameters=string_parameters,
        scale_mesh=scale_mesh,
        grid_elements=grid_elements,
        grid_size=grid_size,
        grid_center=grid_center,
    )

    for i in range(n_params):
        jac[i] = (sdf_fields[i + 1] - sdf_fields[0]) / epsilon

    return jac


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
            "points": ShapeDType(
                shape=(abstract_inputs.max_points, 3), dtype="float32"
            ),
            "faces": ShapeDType(shape=(abstract_inputs.max_faces, 3), dtype="float32"),
            "n_points": ShapeDType(shape=(), dtype="int32"),
            "n_faces": ShapeDType(shape=(), dtype="int32"),
        },
    }
