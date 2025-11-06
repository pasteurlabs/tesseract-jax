from typing import Any

import jax.numpy as jnp
from jax.scipy.interpolate import RegularGridInterpolator
from pydantic import BaseModel, Field
from scipy.interpolate import griddata
from tesseract_core.runtime import Array, Differentiable, Float32, Int32, ShapeDType

#
# Schemata
#


class InputSchema(BaseModel):
    """Input schema for hexahedral mesh generation and field interpolation."""

    field_values: Differentiable[
        Array[
            (None, None, None),
            Float32,
        ]
    ] = Field(
        description=("Values defined on a regular grid that are to be differentiated.")
    )
    sizing_field: Array[
        (None, None, None),
        Float32,
    ] = Field(
        description=(
            "Sizing field values defined on a regular grid for mesh adaptation."
        )
    )
    domain_size: tuple[float, float, float] = Field(
        description=("Size of the domain in x, y, z directions.")
    )

    max_points: int = Field(
        default=10000,
        description=("Maximum number of points in the output hex mesh. "),
    )

    max_cells: int = Field(
        default=10000,
        description=("Maximum number of hexahedral cells in the output hex mesh. "),
    )

    max_subdivision_levels: int = Field(
        default=5,
        description=("Maximum number of subdivision levels for the hex mesh. "),
    )


class HexMesh(BaseModel):
    """Hexagonal mesh representation."""

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


class OutputSchema(BaseModel):
    """Output schema for hexahedral mesh generation and field interpolation."""

    mesh: HexMesh = Field(description="Hexagonal mesh representation of the geometry")
    mesh_cell_values: Differentiable[
        Array[
            (None,),
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
    """Create a single HEX8 mesh of a cuboid domain."""
    # Define the 8 corner points of the hexahedron
    points = jnp.array(
        [
            [-Lx / 2, -Ly / 2, -Lz / 2],  # Point 0
            [Lx / 2, -Ly / 2, -Lz / 2],  # Point 1
            [Lx / 2, Ly / 2, -Lz / 2],  # Point 2
            [-Lx / 2, Ly / 2, -Lz / 2],  # Point 3
            [-Lx / 2, -Ly / 2, Lz / 2],  # Point 4
            [Lx / 2, -Ly / 2, Lz / 2],  # Point 5
            [Lx / 2, Ly / 2, Lz / 2],  # Point 6
            [-Lx / 2, Ly / 2, Lz / 2],  # Point 7
        ],
        dtype=jnp.float32,
    )

    # Define the hexahedron cell using the point indices
    hex_cells = jnp.array(
        [
            [0, 1, 2, 3, 4, 5, 6, 7]  # Single HEX8 element
        ],
        dtype=jnp.int32,
    )

    return points, hex_cells


def vectorized_subdivide_hex_mesh(
    hex_cells: jnp.ndarray,  # (n_hex, 8)
    pts_coords: jnp.ndarray,  # (n_points, 3)
    mask: jnp.ndarray,  # (n_hex,) boolean array indicating which hexes to subdivide
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Vectorized subdivision of HEX8 mesh.

    This method introduces duplicates of points that should later be merged.

    Hexahedron is constructed as follows:

          7 -------- 6
         /|         /|
        4 -------- 5 |
        | |        | |
        | 3 -------|-2
        |/         |/
        0 -------- 1

    Axis orientation:

        z  y
        | /
        |/____ x

    """
    n_hex_new = mask.sum()
    n_new_pts = (8 * 8) * n_hex_new  # 8 corners per new hex, 8 new hexes per old hex

    new_pts_coords = jnp.zeros((n_new_pts, 3), dtype=pts_coords.dtype)
    new_hex_cells = jnp.zeros((n_hex_new * 8, 8), dtype=hex_cells.dtype)

    voxel_sizes = jnp.abs(
        pts_coords[hex_cells[mask, 6]] - pts_coords[hex_cells[mask, 0]]
    )

    center_points = jnp.mean(pts_coords[hex_cells[mask]], axis=1)  # (n_hex, 3)
    offsets = jnp.array(
        [
            [-0.25, -0.25, -0.25],
            [0.25, -0.25, -0.25],
            [0.25, 0.25, -0.25],
            [-0.25, 0.25, -0.25],
            [-0.25, -0.25, 0.25],
            [0.25, -0.25, 0.25],
            [0.25, 0.25, 0.25],
            [-0.25, 0.25, 0.25],
        ]
    ).reshape((1, 8, 3)).repeat(voxel_sizes.shape[0], axis=0) * voxel_sizes.reshape(
        (n_hex_new, 1, 3)
    ).repeat(8, axis=1)

    for cell in range(8):
        center = center_points + offsets[:, cell]

        for corner in range(8):
            new_pts_coords = new_pts_coords.at[
                jnp.arange(n_hex_new) * 64 + cell * 8 + corner
            ].set(center + offsets[:, corner])

            new_hex_cells = new_hex_cells.at[
                jnp.arange(n_hex_new) * 8 + cell, corner
            ].set(jnp.arange(n_hex_new) * 64 + cell * 8 + corner)

    def reindex_and_mask(
        coords: jnp.ndarray, cells: jnp.ndarray, keep_mask: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Reindex points and cells based on mask."""
        # map mask to points
        point_mask = jnp.zeros(coords.shape[0], dtype=jnp.float32)
        point_mask = point_mask.at[cells.flatten()].add(keep_mask.repeat(8))

        # Reindex new points and cells based on mask
        index_offset = jnp.cumsum(jnp.logical_not(point_mask))
        cells = cells - index_offset.at[cells.flatten()].get().reshape(cells.shape)

        # apply mask to keep only subdivided hexes
        coords = coords.at[point_mask > 0].get()
        cells = cells.at[keep_mask].get()

        return coords, cells

    # new_pts_coords, new_hex_cells = reindex_and_mask(
    #     new_pts_coords, new_hex_cells, mask.repeat(8)
    # )
    old_pts_coords, old_hex_cells = reindex_and_mask(
        pts_coords, hex_cells, jnp.logical_not(mask)
    )

    old_hex_cells = old_hex_cells + new_pts_coords.shape[0]

    combined_pts_coords = jnp.vstack([new_pts_coords, old_pts_coords])
    combined_hex_cells = jnp.vstack([new_hex_cells, old_hex_cells])

    return combined_pts_coords, combined_hex_cells


def remove_duplicate_points(
    pts_coords: jnp.ndarray, hex_cells: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Remove duplicate points from the mesh and update hex cell indices."""
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
        hex_cells: Initial hexahedral cells.
        pts_coords: Initial points coordinates.
        sizing_field: Sizing field values on a regular grid.
        levels: Maximum number of subdivision levels.
        Lx: Length of the domain in x direction.
        Ly: Length of the domain in y direction.
        Lz: Length of the domain in z direction.

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

    for i in range(levels):
        voxel_sizes = jnp.max(
            jnp.abs(pts_coords[hex_cells[:, 6]] - pts_coords[hex_cells[:, 0]]), axis=1
        )
        voxel_center_points = jnp.mean(pts_coords[hex_cells], axis=1)
        sizing_values = interpolator(voxel_center_points)
        subdivision_mask = voxel_sizes > sizing_values

        if not jnp.any(subdivision_mask):
            print(f"No more subdivisions needed at level {i}.")
            break

        pts_coords, hex_cells = vectorized_subdivide_hex_mesh(
            hex_cells, pts_coords, subdivision_mask
        )

        pts_coords, hex_cells = remove_duplicate_points(pts_coords, hex_cells)

    return pts_coords, hex_cells


# @lru_cache(maxsize=1)
def generate_mesh(
    Lx: float,
    Ly: float,
    Lz: float,
    sizing_field: jnp.ndarray,
    max_levels: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Generate adapted HEX8 mesh based on sizing field.

    Args:
        Lx: Length of the domain in x direction.
        Ly: Length of the domain in y direction.
        Lz: Length of the domain in z direction.
        sizing_field: Sizing field values on a regular grid.
        max_levels: Maximum number of subdivision levels.

    Returns:
        points: (n_points, 3) array of vertex positions.
        hex_cells: (n_hex, 8) array of hexahedron cell indices.
    """
    initial_pts, initial_hex_cells = create_single_hex(Lx, Ly, Lz)

    pts, cells = recursive_subdivide_hex_mesh(
        initial_hex_cells,
        initial_pts,
        sizing_field,
        levels=max_levels,
        Lx=Lx,
        Ly=Ly,
        Lz=Lz,
    )

    return pts, cells


def apply(inputs: InputSchema) -> OutputSchema:
    """Generate hexahedral mesh and interpolate field values onto cell centers.

    Args:
        inputs: InputSchema, inputs to the function.

    Returns:
        OutputSchema, outputs of the function.
    """
    Lx = inputs.domain_size[0]
    Ly = inputs.domain_size[1]
    Lz = inputs.domain_size[2]
    pts, cells = generate_mesh(
        Lx=Lx,
        Ly=Ly,
        Lz=Lz,
        sizing_field=inputs.sizing_field,
        max_levels=inputs.max_subdivision_levels,
    )

    pts_padded = jnp.zeros((inputs.max_points, 3), dtype=pts.dtype)
    pts_padded = pts_padded.at[: pts.shape[0], :].set(pts)
    cells_padded = jnp.zeros((inputs.max_cells, 8), dtype=cells.dtype)
    cells_padded = cells_padded.at[: cells.shape[0], :].set(cells)

    xs = jnp.linspace(-Lx / 2, Lx / 2, inputs.field_values.shape[0])
    ys = jnp.linspace(-Ly / 2, Ly / 2, inputs.field_values.shape[1])
    zs = jnp.linspace(-Lz / 2, Lz / 2, inputs.field_values.shape[2])

    interpolator = RegularGridInterpolator(
        (xs, ys, zs),
        inputs.field_values,
        method="linear",
        bounds_error=False,
        fill_value=-1,
    )

    cell_centers = jnp.mean(pts[cells], axis=1)
    cell_values = interpolator(cell_centers)

    cell_values_padded = jnp.zeros((inputs.max_cells,), dtype=cell_values.dtype)
    cell_values_padded = cell_values_padded.at[: cell_values.shape[0]].set(cell_values)

    return OutputSchema(
        mesh=HexMesh(
            points=pts_padded.astype(jnp.float32),
            faces=cells_padded.astype(jnp.int32),
            n_points=pts.shape[0],
            n_faces=cells.shape[0],
        ),
        mesh_cell_values=cell_values_padded,
    )


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
) -> dict[str, Any]:
    """Compute vector-Jacobian product for the apply function.

    Our cotangent gradient is defined on the cells centers
    we need to backpropagate it to the field values defined on the regular grid
    this can be done using interpolation
    We need to have the mesh cell center positions here, so instead of recomputing the mesh,
    lets use the cached mesh from the last forward pass
    print(generate_mesh.cache_info())

    Args:
        inputs: InputSchema, inputs to the apply function.
        vjp_inputs: set of input variable names for which to compute the VJP.
        vjp_outputs: set of output variable names for which the cotangent vector is provided.
        cotangent_vector: dict mapping output variable names to their cotangent vectors.

    Returns:
        dict mapping input variable names to their VJP results.
    """
    assert vjp_inputs == {"field_values"}
    assert vjp_outputs == {"mesh_cell_values"}

    Lx = inputs.domain_size[0]
    Ly = inputs.domain_size[1]
    Lz = inputs.domain_size[2]

    pts, cells = generate_mesh(
        Lx=Lx,
        Ly=Ly,
        Lz=Lz,
        sizing_field=inputs.sizing_field,
        max_levels=inputs.max_subdivision_levels,
    )

    cell_centers = jnp.mean(pts[cells], axis=1)

    xs = jnp.linspace(-Lx / 2, Lx / 2, inputs.field_values.shape[0])
    ys = jnp.linspace(-Ly / 2, Ly / 2, inputs.field_values.shape[1])
    zs = jnp.linspace(-Lz / 2, Lz / 2, inputs.field_values.shape[2])
    xs, ys, zs = jnp.meshgrid(xs, ys, zs, indexing="ij")

    field_cotangent_vector = griddata(
        cell_centers,
        cotangent_vector["mesh_cell_values"][: cells.shape[0]],
        (xs, ys, zs),
        method="nearest",
    )

    return {"field_values": jnp.array(field_cotangent_vector).astype(jnp.float32)}


def abstract_eval(abstract_inputs: InputSchema) -> dict[str, ShapeDType]:
    """Calculate output shape of apply from the shape of its inputs."""
    return {
        "mesh_cell_values": ShapeDType(
            shape=(abstract_inputs.max_cells,),
            dtype="float32",
        ),
        "mesh": {
            "points": ShapeDType(
                shape=(abstract_inputs.max_points, 3), dtype="float32"
            ),
            "faces": ShapeDType(shape=(abstract_inputs.max_cells, 8), dtype="int32"),
            "n_points": ShapeDType(shape=(), dtype="int32"),
            "n_faces": ShapeDType(shape=(), dtype="int32"),
        },
    }
