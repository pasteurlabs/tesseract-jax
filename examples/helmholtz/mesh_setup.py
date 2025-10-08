from math import ceil
import jax.numpy as jnp
from jax_fem.generate_mesh import Mesh, rectangle_mesh, get_meshio_cell_type
import pygmsh
import meshio

# mesh = create_circle_mesh(radius=1.0, mesh_size=0.1)

# # Save the mesh
# meshio.write("circle.vtk", mesh)

# # Print mesh info
# print(f"Number of points: {len(mesh.points)}")
# print(f"Number of cells: {len(mesh.cells[0].data)}")

# for cell in mesh.cells:
#     print(f"Cell type: {cell.type}")
# print(mesh.cells[0].data.shape)
# print(mesh.points.shape)
# plt.figure()
# plt.triplot(mesh.points[:, 0], mesh.points[:, 1], mesh.cells[1].data)
# plt.axis('equal')
# plt.show()

def create_circle_meshio(radius=1.0, mesh_size=0.1) -> meshio.Mesh:
    """Create a 2D circle mesh using pygmsh
    
    Args:
        mesh_size: target edge length of the triangular elements the mesh generator aims for.
    """
    with pygmsh.geo.Geometry() as geom:
        # Create a circle
        circle = geom.add_circle([0.0, 0.0, 0.0], radius, mesh_size=mesh_size)
        # Add the surface
        geom.add_plane_surface(circle.curve_loop)
        # Generate mesh
        mesh = geom.generate_mesh()
    
    return mesh

def create_rectangular_mesh(Lx, Ly, c, f_max, ppw) -> tuple:
    dx = c / (f_max * ppw)
    print(dx)

    # Create mesh
    ele_type = 'QUAD4'
    cell_type = get_meshio_cell_type(ele_type)

    Nx, Ny = ceil(Lx / dx), ceil(Ly / dx)      # mesh resolution
    meshio_mesh = rectangle_mesh(Nx, Ny, Lx, Ly)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

    # define boundary locations
    def left(point):
        return jnp.isclose(point[0], 0., atol=1e-5)

    def right(point):
        return jnp.isclose(point[0], Lx, atol=1e-5)

    def bottom(point):
        return jnp.isclose(point[1], 0., atol=1e-5)

    def top(point):
        return jnp.isclose(point[1], Ly, atol=1e-5)

    # def dirichlet_val_left(point):
    #     return 0.

    # def dirichlet_val_right(point):
    #     return 0.

    # location_fns1 = [left, right]
    # value_fns = [dirichlet_val_left, dirichlet_val_right]
    # vecs = [0, 0]
    # dirichlet_bc_info = [location_fns1, vecs, value_fns]

    location_fns2 = [left, right, bottom, top]

    return (mesh, location_fns2, ele_type)

# [copied from jax-fem application examples]
# A little program to find orientation of 3 points
# Coplied from https://www.geeksforgeeks.org/orientation-3-ordered-points/
class Point:
    # to store the x and y coordinates of a point
    def __init__(self, x, y):
        self.x = x
        self.y = y
 
def orientation(p1, p2, p3):
    # To find the orientation of  an ordered triplet (p1,p2,p3) function returns the following values:
    # 0 : Collinear points
    # 1 : Clockwise points
    # 2 : Counterclockwise
    val = (float(p2.y - p1.y) * (p3.x - p2.x)) - (float(p2.x - p1.x) * (p3.y - p2.y))
    if (val > 0):
        # Clockwise orientation
        return 1
    elif (val < 0):
        # Counterclockwise orientation
        return 2
    else:
        # Collinear orientation
        return 0
    
def transform_cells(cells, points, ele_type):
    """FEniCS triangular mesh is not always counter-clockwise. We need to fix it.
    """
    import numpy as np

    new_cells = []
    for cell in cells:
        pts = points[cell[:3]]
        p1 = Point(pts[0, 0], pts[0, 1])
        p2 = Point(pts[1, 0], pts[1, 1])
        p3 = Point(pts[2, 0], pts[2, 1])
         
        o = orientation(p1, p2, p3)
         
        if (o == 0):
            print(f"Linear")
            print(f"Can't be linear, somethign wrong!")
            exit()
        elif (o == 1):
            # print(f"Clockwise")
            if ele_type == 'TRI3':
                new_celll = cell[[0, 2, 1]]
            elif ele_type == 'TRI6':
                new_celll = cell[[0, 2, 1, 5, 4, 3]]
            else:
                print(f"Wrong element type, can't be transformed")
                exit()
            new_cells.append(new_celll)
        else:
            # print(f"CounterClockwise")
            new_cells.append(cell)

    return np.stack(new_cells)

def create_circular_mesh(radius, c, f_max, ppw) -> tuple:    
    dx = c / (f_max * ppw)
    print(dx)

    # Create mesh
    ele_type = 'TRI3'
    cell_type = get_meshio_cell_type(ele_type)

    meshio_mesh = create_circle_meshio(radius=radius, mesh_size=dx)
    points = meshio_mesh.points[:, 0:2]
    cells = meshio_mesh.cells_dict[cell_type]
    cells = transform_cells(cells, points, ele_type)
    mesh = Mesh(points, cells)

    # Split by angle
    def top_right_quadrant(point):
        """Top-right quadrant: 0° to 90°"""
        r = jnp.sqrt(point[0]**2 + point[1]**2)
        angle = jnp.arctan2(point[1], point[0])  # -π to π
        return jnp.logical_and(
            jnp.isclose(r, radius, atol=1e-5),
            jnp.logical_and(angle >= 0, angle <= jnp.pi/2)
        )
    
    def top_left_quadrant(point):
        """Top-left quadrant: 90° to 180°"""
        r = jnp.sqrt(point[0]**2 + point[1]**2)
        angle = jnp.arctan2(point[1], point[0])
        return jnp.logical_and(
            jnp.isclose(r, radius, atol=1e-5),
            jnp.logical_and(angle > jnp.pi/2, angle <= jnp.pi)
        )
    
    def bottom_left_quadrant(point):
        """Bottom-left quadrant: 180° to 270°"""
        r = jnp.sqrt(point[0]**2 + point[1]**2)
        angle = jnp.arctan2(point[1], point[0])
        return jnp.logical_and(
            jnp.isclose(r, radius, atol=1e-5),
            jnp.logical_and(angle > -jnp.pi, angle <= -jnp.pi/2)
        )
    
    def bottom_right_quadrant(point):
        """Bottom-right quadrant: 270° to 360°"""
        r = jnp.sqrt(point[0]**2 + point[1]**2)
        angle = jnp.arctan2(point[1], point[0])
        return jnp.logical_and(
            jnp.isclose(r, radius, atol=1e-5),
            jnp.logical_and(angle > -jnp.pi/2, angle < 0)
        )

    location_fns2 = [top_right_quadrant, top_left_quadrant, bottom_left_quadrant, bottom_right_quadrant]

    return (mesh, location_fns2, ele_type)