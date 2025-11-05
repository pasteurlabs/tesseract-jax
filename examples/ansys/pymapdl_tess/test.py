import os

import numpy as np
from dotenv import load_dotenv
from tesseract_core import Tesseract

load_dotenv()

host = os.getenv("MAPDL_HOST")
if host is None:
    raise ValueError("Unable to read $MAPDL_HOST from the environment.")
port = os.getenv("MAPDL_PORT")
if port is None:
    raise ValueError("Unable to read $MAPDL_PORT from the environment.")


tess_simp_compliance = Tesseract.from_tesseract_api(
    "tesseract_pymapdl/tess_simp_compiance/tesseract_api.py"
)
Lx, Ly, Lz = 3, 2, 1
Nx, Ny, Nz = 60, 40, 20
# Nx, Ny, Nz = 6, 4, 2

n_elem = Nx * Ny * Nz
# Create a test density field varying from 0 to 1
rho = (np.arange(0, n_elem, 1) / n_elem).reshape((n_elem, 1))
rho = 0.5 * np.ones((n_elem, 1))


inputs = {
    "host": host,
    "port": port,
    "rho": rho,
    "Lx": Lx,
    "Ly": Ly,
    "Lz": Lz,
    "Nx": Nx,
    "Ny": Ny,
    "Nz": Nz,
    "E0": 1.0,
    "rho_min": 1e-6,
    "total_force": 0.1,
    "log_level": "DEBUG",
    "vtk_output": "mesh_density.vtk",
}

outputs = tess_simp_compliance.apply(inputs)

# Verify relationship between compliance and strain energy
# For static analysis: Total Strain Energy = 0.5 * Compliance
strain_energy = outputs["strain_energy"]
compliance = outputs["compliance"]
total_strain_energy = np.sum(strain_energy)
print(f"\nCompliance: {compliance:.6e}")
print(f"Total Strain Energy: {total_strain_energy:.6e}")
print(f"0.5 * Compliance: {0.5 * compliance:.6e}")
print(f"Ratio (should be ~1.0): {total_strain_energy / (0.5 * compliance):.6f}")

# Finite difference check
num_tests = 0  # set to 0 if you don't want to run this check
FD_delta = 1.0e-3
f0 = outputs["compliance"]
sensitivity = outputs["sensitivity"]
FD_sensitivity = 0 * sensitivity
for i in range(num_tests):
    print(i)
    inputs["rho"][i] += FD_delta
    outputs = tess_simp_compliance.apply(inputs)
    fupp = outputs["compliance"]
    FD_sensitivity[i] = (fupp - f0) / FD_delta
    inputs["rho"][i] -= FD_delta


if num_tests > 0:
    sens = sensitivity[0:num_tests]
    FD_sens = FD_sensitivity[0:num_tests]
    print(sens)
    print(FD_sens)
    errors = sens - FD_sens
    print(errors)
    rel_abs_error = np.abs(errors / sens)
    print(rel_abs_error)
    print(f"Should be under 1e-5: {np.max(rel_abs_error)}")
