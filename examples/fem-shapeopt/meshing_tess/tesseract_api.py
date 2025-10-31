from typing import Any

import trimesh
from pydantic import BaseModel, Field
from jax.scipy.interpolate import RegularGridInterpolator
from tesseract_core.runtime import Array, Differentiable, Float32, Int32, ShapeDType
import jax.numpy as jnp

#
# Schemata
#


class InputSchema(BaseModel):
    field_values : Differentiable[
        Array[
            (None, None, None),
            Float32,
        ]
    ] = Field(
        description=(
            "Values defined on a regular grid that are to be differentiated."
        )
    )

    sizing_field : Differentiable[
        Array[
            (None, None, None),
            Float32,
        ]
    ] = Field(
        description=(
            "Sizing field values defined on a regular grid for mesh adaptation."
        )
    )

    Lx: float = Field(
        default=60.0,
        description=("Length of the domain in the x direction. "),
    )
    Ly: float = Field(
        default=30.0,
        description=("Length of the domain in the y direction. "),
    )
    Lz: float = Field(
        default=30.0,
        description=("Length of the domain in the z direction. "),
    )

    max_points : int = Field(
        default=10000,
        description=("Maximum number of points in the output hex mesh. "),
    )

    max_cells : int = Field(
        default=10000,
        description=("Maximum number of hexahedral cells in the output hex mesh. "),
    )

    max_subdivision_levels : int = Field(
        default=5,
        description=("Maximum number of subdivision levels for the hex mesh. "),
    )


class HexMesh(BaseModel):
    points: Array[(None, 3), Float32] = Field(description="Array of vertex positions.")
    faces: Array[(None, 8), Float32] = Field(
        description="Array of hexahedral faces defined by indices into the points array."
    )
    n_points: Int32 = Field(
        default=0, description="Number of valid points in the points array."
    )
    n_faces: Int32 = Field(
        default=0, description="Number of valid faces in the faces array."
    )


class OutputSchema(BaseModel):
    mesh: HexMesh = Field(
        description="Hexagonal mesh representation of the geometry"
    )
    mesh_cell_values: Differentiable[
        Array[
            (None, None,),
            Float32,
        ]
    ] = Field(description="Cell-centered values defined on the hexahedral mesh.")

#
# Helper functions
#
def create_single_hex(
    Lx: float,
    Ly: float,
    Lz: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """ 
    Create a single HEX8 mesh of a cuboid domain.
    """

    # Define the 8 corner points of the hexahedron
    points = jnp.array([
        [-Lx / 2, -Ly / 2, -Lz / 2],  # Point 0
        [ Lx / 2, -Ly / 2, -Lz / 2],  # Point 1
        [ Lx / 2,  Ly / 2, -Lz / 2],  # Point 2
        [-Lx / 2,  Ly / 2, -Lz / 2],  # Point 3
        [-Lx / 2, -Ly / 2,  Lz / 2],  # Point 4
        [ Lx / 2, -Ly / 2,  Lz / 2],  # Point 5
        [ Lx / 2,  Ly / 2,  Lz / 2],  # Point 6
        [-Lx / 2,  Ly / 2,  Lz / 2],  # Point 7
    ], dtype=jnp.float32)

    # Define the hexahedron cell using the point indices
    hex_cells = jnp.array([
        [0, 1, 2, 3, 4, 5, 6, 7]  # Single HEX8 element
    ], dtype=jnp.int32)

    return points, hex_cells

def vectorized_subdivide_hex_mesh(
    hex_cells: jnp.ndarray, # (n_hex, 8)
    pts_coords: jnp.ndarray, # (n_points, 3)
    mask : jnp.ndarray # (n_hex,) boolean array indicating which hexes to subdivide
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Vectorized subdivision of HEX8 mesh.

    This method introduces duplicates of points that should later be merged.

    Hexahedron is constructed as follows:
             3 -------- 2
         /|         /|
        7 -------- 6 |
        | |        | |
        | 0 -------|-1
        |/         |/
        4 -------- 5
    
    Axis orientation:
        y
        |
        |____ x
       /
      /
     z

    """

    n_hex = hex_cells.shape[0]
    n_new_pts = (8 * 8) * n_hex  # 8 corners per new hex, 8 new hexes per old hex

    new_pts_coords = jnp.zeros((n_new_pts, 3), dtype=pts_coords.dtype)
    new_hex_cells = jnp.zeros((n_hex * 8, 8), dtype=hex_cells.dtype)

    voxel_sizes = jnp.abs(pts_coords[hex_cells[:, 6]] - pts_coords[hex_cells[:, 0]])
    
    center_points = jnp.mean(pts_coords[hex_cells], axis=1)  # (n_hex, 3)
    offsets = jnp.array([
        [-0.25, -0.25, -0.25],
        [0.25, -0.25, -0.25],
        [0.25, 0.25, -0.25],
        [-0.25, 0.25, -0.25],
        [-0.25, -0.25, 0.25],
        [0.25, -0.25, 0.25],
        [0.25, 0.25, 0.25],
        [-0.25, 0.25, 0.25],
    ]).reshape((1, 8, 3)).repeat(voxel_sizes.shape[0], axis=0) * voxel_sizes.reshape((n_hex, 1, 3)).repeat(8, axis=1)

    for cell in range(8):
        center = center_points + offsets[:, cell]

        for corner in range(8):
            new_pts_coords = new_pts_coords.at[
                jnp.arange(n_hex) * 64 + cell * 8 + corner
            ].set(
                center - offsets[:, corner]
            )

            new_hex_cells = new_hex_cells.at[
                jnp.arange(n_hex) * 8 + cell, corner
            ].set(
                jnp.arange(n_hex) * 64 + cell * 8 + corner
            )

    def reindex_and_mask(
        coords: jnp.ndarray,
        cells: jnp.ndarray,
        keep_mask: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Reindex points and cells based on mask.
        """
        # map mask to points
        point_mask = jnp.zeros(coords.shape[0], dtype=jnp.float32)
        point_mask = point_mask.at[cells.flatten()].add(keep_mask.repeat(8))
        # Reindex new points and cells based on mask
        index_offset = jnp.cumsum(jnp.logical_not(point_mask))
        cells = cells - index_offset.at[cells.flatten()].get().reshape(cells.shape)

        # apply mask to keep only subdivided hexes
        coords = coords.at[keep_mask.repeat(8)].get()
        cells = cells.at[keep_mask].get()

        return coords, cells

    new_pts_coords, new_hex_cells = reindex_and_mask(
        new_pts_coords, new_hex_cells, mask.repeat(8)
    )
    # TODO: This does not work when duplicate edges are removed inbetween subdivisions
    old_pts_coords, old_hex_cells = reindex_and_mask(
        pts_coords, hex_cells, jnp.logical_not(mask)
    )

    old_hex_cells = old_hex_cells + new_pts_coords.shape[0]

    combined_pts_coords = jnp.vstack([new_pts_coords, old_pts_coords])
    combined_hex_cells = jnp.vstack([new_hex_cells, old_hex_cells])

    return combined_pts_coords, combined_hex_cells


def remove_duplicate_points(
    pts_coords: jnp.ndarray,
    hex_cells: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Remove duplicate points from the mesh and update hex cell indices.
    """

    unique_pts, inverse_indices = jnp.unique(pts_coords, axis=0, return_inverse=True)
    updated_hex_cells = inverse_indices[hex_cells]

    return unique_pts, updated_hex_cells

def recursive_subdivide_hex_mesh(
    hex_cells: jnp.ndarray,
    pts_coords: jnp.ndarray,
    sizing_field: jnp.ndarray,
    levels: int,
    Lx: float,
    Ly: float,
    Lz: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Recursively (unrolled) subdivide HEX8 mesh.

    Args:
        hex_cells: (n_hex, 8) array of hexahedron cell indices.
        pts_coords: (n_points, 3) array of point coordinates.
        sizing_field: Sizing field values at each point. 
        levels: Number of subdivision levels.

    Returns:
        Subdivided points and hex cells.
    """

    # lets build the kd-tree for fast nearest neighbor search
    xs = jnp.linspace(-Lx / 2, Lx / 2, sizing_field.shape[0])
    ys = jnp.linspace(-Ly / 2, Ly / 2, sizing_field.shape[1])
    zs = jnp.linspace(-Lz / 2, Lz / 2, sizing_field.shape[2])

    interpolator = RegularGridInterpolator(
        (xs, ys, zs), sizing_field, method="linear", bounds_error=False, fill_value=-1
    )

    for _ in range(levels):

        voxel_sizes = jnp.max(jnp.abs(pts_coords[hex_cells[:, 6]] - pts_coords[hex_cells[:, 0]]), axis=1)
        voxel_center_points = jnp.mean(pts_coords[hex_cells], axis=1)
        sizing_values = interpolator(voxel_center_points)
        subdivision_mask = voxel_sizes > sizing_values

        pts_coords, hex_cells = vectorized_subdivide_hex_mesh(
            hex_cells, pts_coords, subdivision_mask
        )
       

    pts_coords, hex_cells = remove_duplicate_points(pts_coords, hex_cells)

    return pts_coords, hex_cells


def apply(inputs: InputSchema) -> OutputSchema:

    initial_pts, initial_hex_cells = create_single_hex(inputs.Lx, inputs.Ly, inputs.Lz)

    pts_2, hex_cells_2 = recursive_subdivide_hex_mesh(
        initial_hex_cells, initial_pts, inputs.sizing_field, levels=inputs.max_subdivision_levels, Lx=inputs.Lx, Ly=inputs.Ly, Lz=inputs.Lz
    )

    return OutputSchema(
        
    )


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
):
    assert vjp_inputs == {"bar_params"}
    assert vjp_outputs == {"sdf"}

    jac = jac_sdf_wrt_params(
        inputs.bar_params,
        radius=inputs.bar_radius,
        Lx=inputs.Lx,
        Ly=inputs.Ly,
        Lz=inputs.Lz,
        Nx=inputs.Nx,
        Ny=inputs.Ny,
        Nz=inputs.Nz,
        epsilon=inputs.epsilon,
    )
    if inputs.normalize_jacobian:
        n_elements = inputs.Nx * inputs.Ny * inputs.Nz
        jac = jac / n_elements
    # Reduce the cotangent vector to the shape of the Jacobian, to compute VJP by hand
    vjp = np.einsum("ijklmn,lmn->ijk", jac, cotangent_vector["sdf"]).astype(np.float32)
    if inputs.normalize_vjp:
        vjp_std = np.std(vjp)
        if vjp_std > 0:
            vjp = vjp / vjp_std

    return {"bar_params": vjp}


def abstract_eval(abstract_inputs):
    """Calculate output shape of apply from the shape of its inputs."""
    return {
        "sdf": ShapeDType(
            shape=(abstract_inputs.Nx, abstract_inputs.Ny, abstract_inputs.Nz),
            dtype="float32",
        ),
        "mesh": {
            "points": ShapeDType(shape=(N_POINTS, 3), dtype="float32"),
            "faces": ShapeDType(shape=(N_FACES, 3), dtype="float32"),
            "n_points": ShapeDType(shape=(), dtype="int32"),
            "n_faces": ShapeDType(shape=(), dtype="int32"),
        },
    }
