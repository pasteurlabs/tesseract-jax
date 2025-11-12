"""
/usr/ansys_inc/v241/ansys/bin/ansys241 -port 50005  -grpc
"""
import os

import numpy as np
from tesseract_core import Tesseract


# load the MAPDL server's address from your environment
host = os.getenv("MAPDL_HOST")
if host is None:
    raise ValueError("Unable to read $MAPDL_HOST from the environment. " + \
        "Use 'export MAPDL_HOST=X.X.X.X' for local IP address of your MAPDL Instance.")
port = os.getenv("MAPDL_PORT")
if port is None:
    raise ValueError("Unable to read $MAPDL_PORT from the environment. " + \
        "Use 'export MAPDL_PORT=X' for the port of your MAPDL Instance.")


# Initialize Tesseract from api (for testing)
try:
    tess_simp_compliance = Tesseract.from_tesseract_api(
        "./tesseract_api.py"
    )
except RuntimeError as e:
    raise RuntimeError("Unable to load tesseract from api. " \
        "Ensure that you have installed the build requirements using 'pip install -r tesseract_requirements.txt'")
    
# TODO
# we should probably create some re-usable mesh constructors...
# create a simple Hex mesh using Pyvista
import pyvista as pv
Lx, Ly, Lz = 3, 2, 1
# Nx, Ny, Nz = 60, 40, 20
Nx, Ny, Nz = 6, 4, 2
grid = pv.ImageData(
    dimensions=np.array((Nx, Ny, Nz)) + 1,
    #origin=(-Lx / 2, -Ly / 2, -Lz / 2), # The bottom left corner of the data set
    origin=(0, 0, 0), # TODO
    spacing=(Lx / Nx, Ly / Ny, Lz / Nz), # These are the cell sizes along each axis
)
# repeated casts will eventually expose cell_connectivitiy
mesh = grid.cast_to_structured_grid().cast_to_explicit_structured_grid().cast_to_unstructured_grid()

from tesseract_api import HexMesh
hex_mesh = HexMesh(
    points=mesh.points,
    faces=mesh.cell_connectivity.reshape((Nx * Ny * Nz, 8)),
    n_points=mesh.points.shape[0],
    n_faces=Nx * Ny * Nz,
)

# dirichlet condition
on_lhs = mesh.points[:, 0] <= -Lx / 2
# TODO should this be an n_node vector?
# dirichlet_indices = np.where(on_lhs)[0]
# dirichlet_mask = np.zeros(hex_mesh.n_points)
# dirichlet_mask[dirichlet_indices] = 1
dirichlet_mask = np.where(on_lhs)[0]
dirichlet_values = np.zeros((dirichlet_mask.shape[0]))

# von Neumann condition
x_lim = Lx/2
y_min = 0 # -Ly/2 # TODO
y_max = y_min + 0.2 * Ly
z_min = 0 # 0.0 - 0.1 * Lz
z_max = z_min + 0.2 * Lz
von_neumann = np.logical_and(
    mesh.points[:, 0] >= x_lim,
    np.logical_and(mesh.points[:,1] >= y_min, mesh.points[:,1] <= y_max),
    np.logical_and(mesh.points[:,2] >= z_min, mesh.points[:,2] <= z_max),
)
# TODO should this be an n_node array?
von_neumann_mask = np.where(von_neumann)[0]
von_neumann_values = (0, 0.0, 0.1 / len(von_neumann_mask)) + np.zeros((von_neumann_mask.shape[0], 3))

# Create a test density field varying from 0 to 1
n_elem = Nx * Ny * Nz
rho = (np.arange(0, n_elem, 1) / n_elem).reshape((n_elem, 1))
rho = 0.5 * np.ones((n_elem, 1))
 
 
inputs = {
    "dirichlet_mask": dirichlet_mask,
    "dirichlet_values": dirichlet_values,
    "van_neumann_mask": von_neumann_mask,
    "van_neumann_values": von_neumann_values,
    "hex_mesh": dict(hex_mesh),
    "host": host,
    "port": port,
    "rho": rho,
    "E0": 1.0,
    "rho_min": 1e-6,
    "log_level": "DEBUG",
    "vtk_output": "mesh_density.vtk",
}

# cells = np.array([8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 9, 10, 11, 12, 13, 14, 15])
cells = np.concatenate([np.array([[8]] * hex_mesh.n_faces), hex_mesh.faces], 1)
print(cells)
celltypes = np.full(hex_mesh.n_faces, pv.CellType.HEXAHEDRON, dtype=np.uint8)
mesh2 = pv.UnstructuredGrid(cells.ravel(), celltypes, hex_mesh.points)
print(mesh2 == mesh)
print(mesh.cell_connectivity)
print(mesh2.cell_connectivity)

#outputs = tess_simp_compliance.apply(inputs)
#
## Verify relationship between compliance and strain energy
## For static analysis: Total Strain Energy = 0.5 * Compliance
#strain_energy = outputs["strain_energy"]
#compliance = outputs["compliance"]
#total_strain_energy = np.sum(strain_energy)
#print(f"\nCompliance: {compliance:.6e}")
#print(f"Total Strain Energy: {total_strain_energy:.6e}")
#print(f"0.5 * Compliance: {0.5 * compliance:.6e}")
#print(f"Ratio (should be ~1.0): {total_strain_energy / (0.5 * compliance):.6f}")

# # Finite difference check
# num_tests = 0  # set to 0 if you don't want to run this check
# FD_delta = 1.0e-3
# f0 = outputs["compliance"]
# sensitivity = outputs["sensitivity"]
# FD_sensitivity = 0 * sensitivity
# for i in range(num_tests):
#     print(i)
#     inputs["rho"][i] += FD_delta
#     outputs = tess_simp_compliance.apply(inputs)
#     fupp = outputs["compliance"]
#     FD_sensitivity[i] = (fupp - f0) / FD_delta
#     inputs["rho"][i] -= FD_delta
# 
# 
# if num_tests > 0:
#     sens = sensitivity[0:num_tests]
#     FD_sens = FD_sensitivity[0:num_tests]
#     print(sens)
#     print(FD_sens)
#     errors = sens - FD_sens
#     print(errors)
#     rel_abs_error = np.abs(errors / sens)
#     print(rel_abs_error)
#     print(f"Should be under 1e-5: {np.max(rel_abs_error)}")
