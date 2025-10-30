import jax.numpy as jnp
import jax
from jax.scipy.interpolate import RegularGridInterpolator

def create_hex(
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
    """

    n_hex = hex_cells.shape[0]
    n_new_pts = (8 * 8) * n_hex  # 8 corners per new hex, 8 new hexes per old hex

    new_pts_coords = jnp.zeros((n_new_pts, 3), dtype=pts_coords.dtype)
    new_hex_cells = jnp.zeros((n_hex * 8, 8), dtype=hex_cells.dtype)

    # Hexahedron is constructed as follows:
    #
    #      3 -------- 2
    #      /|         /|
    #     7 -------- 6 |
    #     | |        | |
    #     | 0 -------|-1
    #     |/         |/
    #     4 -------- 5
    # 
    # Axis orientation:
    #     y
    #     |
    #     |____ x
    #    /
    #   /
    #  z

    voxel_sizes = jnp.abs(pts_coords[hex_cells[:, 6]] - pts_coords[hex_cells[:, 0]])
    
    center_points = jnp.mean(pts_coords[hex_cells], axis=1)  # (n_hex, 3)
    # TODO: something wrong on 3rd recursion level here.
    offsets = jnp.array([
        [-0.25, -0.25, -0.25],
        [0.25, -0.25, -0.25],
        [0.25, 0.25, -0.25],
        [-0.25, 0.25, -0.25],
        [-0.25, -0.25, 0.25],
        [0.25, -0.25, 0.25],
        [0.25, 0.25, 0.25],
        [-0.25, 0.25, 0.25],
    ]).repeat(voxel_sizes.shape[0], axis=0) * voxel_sizes.repeat(8, axis=0)
    offsets = offsets.reshape((n_hex, 8, 3))

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


def recursive_subdivide_hex_mesh(
    hex_cells: jnp.ndarray,
    pts_coords: jnp.ndarray,
    sizing_field: jnp.ndarray,
    levels: int,
    Lx: float,
    Ly: float,
    Lz: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Recursively subdivide HEX8 mesh.

    Args:
        hex_cells: (n_hex, 8) array of hexahedron cell indices.
        pts_coords: (n_points, 3) array of point coordinates.
        sizing_field: Sizing field values at each point. 
            Must be 2^levels in each dimension.
        levels: Number of subdivision levels.

    Returns:
        Subdivided points and hex cells.
    """

    # lets build the kd-tree for fast nearest neighbor search
    xs = jnp.linspace(-Lx / 2, Lx / 2, sizing_field.shape[0])
    ys = jnp.linspace(-Ly / 2, Ly / 2, sizing_field.shape[1])
    zs = jnp.linspace(-Lz / 2, Lz / 2, sizing_field.shape[2])

    interpolator = RegularGridInterpolator(
        (xs, ys, zs), sizing_field, method="nearest", bounds_error=False, fill_value=jnp.max(sizing_field)
    )

    for _ in range(levels):
        voxel_sizes = jnp.max(jnp.abs(pts_coords[hex_cells[:, 6]] - pts_coords[hex_cells[:, 0]]), axis=1)
        sizing_values = interpolator(pts_coords[hex_cells].mean(axis=1))
        subdivision_mask = voxel_sizes > sizing_values
        pts_coords, hex_cells = vectorized_subdivide_hex_mesh(
            hex_cells, pts_coords, subdivision_mask
        )

    return pts_coords, hex_cells

def remove_duplicate_points(
    pts_coords: jnp.ndarray,
    hex_cells: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Remove duplicate points from the mesh and update hex cell indices.
    """

    unique_pts, inverse_indices = jnp.unique(pts_coords, axis=0, return_inverse=True)
    updated_hex_cells = inverse_indices[hex_cells]

    return unique_pts, updated_hex_cells

Lx, Ly, Lz = 1.0, 1.0, 1.0

initial_pts, initial_hex_cells = create_hex(Lx, Ly, Lz)

key = jax.random.PRNGKey(0)
sizing = jax.random.uniform(key, shape=(8, 8, 8), minval=0.05, maxval=0.2)

pts, hex_cells = recursive_subdivide_hex_mesh(
    initial_hex_cells, initial_pts, sizing, levels=3, Lx=Lx, Ly=Ly, Lz=Lz
)

print("Initial points: ", initial_pts)
print("Initial hex cells: ", initial_hex_cells)

subdivision_mask = jnp.array([True], dtype=bool)
pts_lvl_1, hex_lvl_1 = vectorized_subdivide_hex_mesh(
    initial_hex_cells, initial_pts, subdivision_mask
)
n_pts_lvl_1 = pts_lvl_1.shape[0]

pts_lvl_1, hex_lvl_1 = remove_duplicate_points(pts_lvl_1, hex_lvl_1)
print(f"Level 1 points reduced from {n_pts_lvl_1} to {pts_lvl_1.shape[0]} after removing duplicates.")

print("Level 1 points shape: ", pts_lvl_1.shape)
print("Level 1 hex cells shape: ", hex_lvl_1.shape)

subdivision_mask = jnp.array([True, False, True, False, True, False, True, False], dtype=bool)
# subdivision_mask = jnp.array([True] * 8, dtype=bool)
pts_lvl_2, hex_lvl_2 = vectorized_subdivide_hex_mesh(
    hex_lvl_1, pts_lvl_1, subdivision_mask
)

n_pts_lvl_2 = pts_lvl_2.shape[0]
# pts_lvl_2, hex_lvl_2 = remove_duplicate_points(pts_lvl_2, hex_lvl_2)
# print(f"Level 2 points reduced from {n_pts_lvl_2} to {pts_lvl_2.shape[0]} after removing duplicates.")

print("Level 2 points shape: ", pts_lvl_2.shape)
print("Level 2 hex cells shape: ", hex_lvl_2.shape)