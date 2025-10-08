# Install: pip install jax-fem jax jaxlib optax matplotlib
from dataclasses import dataclass
import logging
from jax_fem import logger
from math import ceil
from typing import Callable
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import numpy as np
from jax_fem.problem import Problem
from jax_fem.solver import solver, ad_wrapper
from jax_fem.generate_mesh import Mesh, rectangle_mesh, get_meshio_cell_type
from jax_fem.utils import save_sol
import optax
import matplotlib.pyplot as plt

from visualization import animate_pressure_field, plot_pressure_field

# Problem parameters
Lx = 2.0
Ly = 2.0
f_min = 100 # Hz
f_max = 1000 # Hz
c = 343 # speed of sound m/s
Z = 1.5 + 0.3j # impedance

source_center = [1.0, 1.0]

logger.setLevel(logging.NOTSET)

@dataclass
class Source:
    width: float
    center: list[float]
    amplitude: float

    def __init__(self, k_max: float, center: list[float], amplitude: float):
        # Spatial width for Gaussian source
        lambda_min = 2 * jnp.pi / k_max  # Minimum wavelength
        self.width = lambda_min / 4      # Quarter wavelength (conservative)
        self.center = jnp.array(center)
        self.amplitude = amplitude

class AcousticHelmholtz(Problem):
    source_params: Source
    k: float

    def __init__(self, mesh, k, source_params, **kargs):
        """
        mesh: JAX-FEM mesh object
        k: wave number (2π * frequency / speed_of_sound)
        Z: parameters defining boundary impedance
        source_params: parameters defining Gaussian source
        """        
        super().__init__(mesh, **kargs)
        self.k = k
        self.source_params = source_params

    # stiffness matrix
    def get_tensor_map(self):
        """Stiffness matrix solving -div.f(u_grad) -> f is identity for linear problems."""
        return lambda x: x
    
    def get_mass_map(self):
        """Mass matrix for -k² term"""
        def mass_map(u, x):
            # Helmholtz -k² term
            helmholtz_term = -self.k**2 * u

            # Gaussian source term
            r_squared = jnp.sum((x - self.source_params.center)**2)
            source = -self.source_params.amplitude * jnp.exp(-r_squared / (2 * self.source_params.width**2))

            # combine
            return jnp.array(helmholtz_term + source, dtype=jnp.complex128)

        return mass_map    
    
    def get_surface_maps(self):
        """Impedance boundary condition: ∂p/∂n + ikZp = 0"""
            
        def surface_map(u, x, Z):            
            # u here represents test/trial functions on the boundary element
            # This should return the integrand ikZ * p * v
            # where p and v are represented by u
        
            return 1j * self.k * Z * u

        return [surface_map, surface_map, surface_map, surface_map]


    def set_params(self, Z):
        """For inverse problems, the `set_params(params)` method provides the interface for dynamic model parameter updates."""
        
        # Get the quadrature points for each boundary surface
        quad_surface_points0 = self.fes[0].get_physical_surface_quad_points(self.boundary_inds_list[0])
        quad_surface_points1 = self.fes[0].get_physical_surface_quad_points(self.boundary_inds_list[1])
        quad_surface_points2 = self.fes[0].get_physical_surface_quad_points(self.boundary_inds_list[2])
        quad_surface_points3 = self.fes[0].get_physical_surface_quad_points(self.boundary_inds_list[3])

        Z_params0 = Z * np.ones_like(quad_surface_points0[:, :, 0])
        Z_params1 = Z * np.ones_like(quad_surface_points1[:, :, 0])
        Z_params2 = Z * np.ones_like(quad_surface_points2[:, :, 0])
        Z_params3 = Z * np.ones_like(quad_surface_points3[:, :, 0])

        self.internal_vars_surfaces = [[Z_params0], [Z_params1], [Z_params2], [Z_params3]]

def create_rectangular_mesh(Lx, Ly, c, f_max, ppw) -> tuple:
    dx = c / (f_max * ppw)

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

    def dirichlet_val_left(point):
        return 0.

    def dirichlet_val_right(point):
        return 0.

    location_fns1 = [left, right]
    value_fns = [dirichlet_val_left, dirichlet_val_right]
    vecs = [0, 0]
    dirichlet_bc_info = [location_fns1, vecs, value_fns]

    # location_fns2 = [bottom, top]
    location_fns2 = [left, right, bottom, top]

    return (mesh, dirichlet_bc_info, location_fns2, ele_type)

def compute_acoustic_loss(problem, sol_list_pred, sol_list_ref, w_mag, w_phase, w_rel):
    """
    Combined loss for acoustic inverse problems.
    
    Args:
        w_mag: Weight for magnitude error
        w_phase: Weight for phase error (typically smaller)
        w_rel: Weight for relative error term
    """
    p_pred_quad = problem.fes[0].convert_from_dof_to_quad(sol_list_pred[0])
    p_ref_quad = problem.fes[0].convert_from_dof_to_quad(sol_list_ref[0])

    JxW = problem.fes[0].JxW[:, :, None]
    
    # 1. Magnitude error (absolute)
    mag_pred = jnp.abs(p_pred_quad)
    mag_ref = jnp.abs(p_ref_quad)
    mag_error = jnp.sum((mag_pred - mag_ref)**2 * JxW)
    
    # 2. Phase error (wrapped properly)
    phase_diff = jnp.angle(p_pred_quad * jnp.conj(p_ref_quad))
    phase_error = jnp.sum(phase_diff**2 * JxW)
    
    # 3. Relative error (handles different scales)
    relative_error = jnp.abs(p_pred_quad - p_ref_quad)**2 / (jnp.abs(p_ref_quad)**2 + 1e-8)
    rel_loss = jnp.sum(relative_error * JxW)
    
    # Combined loss
    total_loss = w_mag * mag_error + w_phase * phase_error + w_rel * rel_loss
    
    return total_loss

def J(Z_params, fwd_pred, problem, p_ref):
    p_pred = fwd_pred(Z_params)
    data_loss = compute_acoustic_loss(
        problem, p_pred, p_ref,
        w_mag=0.5,
        w_phase=0.5,
        w_rel=0.0 # Helps with scaling
        )

    return data_loss #+ reg_loss

def setup_forward_solver(mesh_data, f_max, c, source_center):
    mesh, dirichlet_bc_info, location_fns2, ele_type = mesh_data

    k_max = 2 * jnp.pi * f_max / c # wave number    
    source = Source(k_max=k_max, center=source_center, amplitude=1000)

    # Create problem instance
    problem = AcousticHelmholtz(
        mesh=mesh, k=k_max, source_params=source,
        vec=1, dim=2, ele_type=ele_type,
        # dirichlet_bc_info=dirichlet_bc_info, 
        location_fns=location_fns2,
        gauss_order=1)    
    
    sol = ad_wrapper(problem)
    return sol, problem

def test_impedance_sensitivity(problem, fwd_pred, Z_true, w_mag=1.0, w_phase=1.0, w_rel=0.0):
    """
    Test how sensitive the pressure field is to real vs imaginary Z.
    """
    import matplotlib.pyplot as plt
    
    # Generate reference solution
    sol_ref = fwd_pred(Z_true)
    
    # Test variation in real part
    Z_real_range = jnp.linspace(0.5, 2.5, 20)
    losses_real = []
    
    for Z_r in Z_real_range:
        Z_test = Z_r + jnp.imag(Z_true) * 1j
        sol_test = fwd_pred(Z_test)
        loss = compute_acoustic_loss(problem, sol_test, sol_ref, w_mag=w_mag, w_phase=w_phase, w_rel=w_rel)
        losses_real.append(loss)
    
    # Test variation in imaginary part
    Z_imag_range = jnp.linspace(-0.5, 1.0, 20)
    losses_imag = []
    
    for Z_i in Z_imag_range:
        Z_test = jnp.real(Z_true) + Z_i * 1j
        sol_test = fwd_pred(Z_test)
        loss = compute_acoustic_loss(problem, sol_test, sol_ref, w_mag=w_mag, w_phase=w_phase, w_rel=w_rel)
        losses_imag.append(loss)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(Z_real_range, losses_real, 'b-', linewidth=2)
    ax1.axvline(x=jnp.real(Z_true), color='r', linestyle='--', label='True value')
    ax1.set_xlabel('Re(Z)')
    ax1.set_ylabel('Loss')
    ax1.set_title('Sensitivity to Real Part')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(Z_imag_range, losses_imag, 'g-', linewidth=2)
    ax2.axvline(x=jnp.imag(Z_true), color='r', linestyle='--', label='True value')
    ax2.set_xlabel('Im(Z)')
    ax2.set_ylabel('Loss')
    ax2.set_title('Sensitivity to Imaginary Part')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Check curvature (second derivative ~ sensitivity)
    real_sensitivity = jnp.std(jnp.diff(losses_real))
    imag_sensitivity = jnp.std(jnp.diff(losses_imag))
    
    print(f"Real part sensitivity: {real_sensitivity:.6e}")
    print(f"Imag part sensitivity: {imag_sensitivity:.6e}")
    print(f"Ratio (real/imag): {real_sensitivity/imag_sensitivity:.2f}")
    
    if real_sensitivity > 3 * imag_sensitivity:
        print("⚠️ WARNING: Loss is much more sensitive to Re(Z) than Im(Z)!")
        print("   This will make imaginary part hard to estimate.")

def optimize_impedance(fwd_pred, problem, measurements, n_iterations, learning_rate):
    """Optimize impedance to match measurements."""    

    # Initialize impedance parameters
    # Could be scalar, or array for spatially-varying impedance    
    Z_init = 3.0 + 0.05j
    
    # Set up optimizer
    optimizer = optax.sgd(learning_rate)
    opt_state = optimizer.init(Z_init)

    Z_opt = Z_init
    
    losses = []
    Z_opt_list = []
    
    for i in range(n_iterations):
        # Compute loss and gradient
        loss_fn = lambda Z: J(Z, fwd_pred, problem, measurements)
        loss, grad = jax.value_and_grad(loss_fn)(Z_opt)
        grad = jnp.real(grad) -1j * jnp.imag(grad)
        
        # Update parameters
        updates, opt_state = optimizer.update(grad, opt_state)
        Z_opt = optax.apply_updates(Z_opt, updates)
        
        losses.append(loss)
        Z_opt_list.append(Z_opt)
        
        # Warning signs
        grad_norm = jnp.linalg.norm(grad)
        if grad_norm > 1e3:
            print("⚠️ WARNING: Gradient explosion!")
        if grad_norm < 1e-8:
            print("⚠️ WARNING: Gradient vanishing!")
            
        if i % 10 == 0:
            print(f"\n*** Iteration {i}: Loss = {loss:.6f}, Z = {Z_opt} ***\n")
    
    return Z_opt, losses, Z_opt_list

def estimate_boundary_impedance(
    Lx, Ly,
    f_max,
    c,
    Z_true,  
    source_center=[1.0, 1.0],
    ppw = 5.0,
    n_iterations = 200,
    learning_rate = 0.1,
):    
    mesh_data = create_rectangular_mesh(Lx, Ly, c, f_max, ppw)

    fwd_pred, problem = setup_forward_solver(mesh_data, f_max, c, source_center)
    measurements = fwd_pred(Z_true)

    # Add noise to measurements
    # measurements = measurements + 0.01 * jax.random.normal(jax.random.PRNGKey(0), measurements.shape)

    # test_impedance_sensitivity(problem, fwd_pred, Z_true=1.5+0.3j)
    
    # Optimize
    Z_opt, losses, Z_opt_list = optimize_impedance(fwd_pred, problem, measurements, n_iterations=n_iterations, learning_rate=learning_rate)

    print(f"\nTrue impedance: {Z}")
    print(f"Estimated impedance: {Z_opt}")

    # Plot convergence
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.semilogy(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Convergence')

    plt.subplot(1, 3, 2)
    plt.plot(np.real(Z_opt_list))
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Z (real)')

    plt.subplot(1, 3, 3)
    plt.plot(np.imag(Z_opt_list))
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Z (imag)')
        
    # Plot pressure field
    # ... visualization code ...
    plt.show()

        
def frequency_to_time_domain(
    Lx, Ly,
    f_min,
    f_max,
    c,
    Z,    
    source_center=[1.0, 1.0],
    ppw = 5.0,
):
    df = 10 # Hz (frequency resolution) TODO: calculate from time requirement?
    k_max = 2 * jnp.pi * f_max / c # wave number

    mesh, dirichlet_bc_info, location_fns2, ele_type = create_rectangular_mesh(Lx, Ly, c, f_max, ppw)

    N_freq = int((f_max - f_min) / df) + 1
    frequencies = np.linspace(f_min, f_max, N_freq)
            
    source = Source(k_max=k_max, center=source_center, amplitude=1000)

    pressure_freq = []

    for i, f in enumerate(frequencies):
        k = 2 * jnp.pi * f / c
        
        # Create problem instance
        problem = AcousticHelmholtz(
            mesh=mesh, k=k, source_params=source,
            vec=1, dim=2, ele_type=ele_type,
            # dirichlet_bc_info=dirichlet_bc_info, 
            location_fns=location_fns2,
            gauss_order=1)        
        problem.set_params(Z)

        sol = solver(problem)
        pressure_freq.append(sol[0][:, 0])

        if i % 10 == 0:
            print(f"#### Solved frequency {i+1}/{len(frequencies)}: {f:.0f} Hz ####")

    T = 1 / df
    dt = 1 / (2 * f_max)

    N_time = int(T / dt)

    # Convert to array [n_nodes, n_freq]
    P_freq = jnp.array(pressure_freq).T

    # Inverse Fourier transform to time domain
    # Need to create full spectrum (positive and negative frequencies)
    P_full = jnp.zeros((P_freq.shape[0], N_time), dtype=jnp.complex128)

    # Fill positive frequencies
    freq_indices = jnp.round(frequencies / df).astype(int)
    P_full = P_full.at[:, freq_indices].set(P_freq)

    # Apply inverse FFT
    p_time = jnp.fft.ifft(P_full, axis=1, n=N_time)

    # Take real part (pressure is real in time domain)
    p_time_real = jnp.real(p_time)

    t = jnp.arange(N_time) * dt

    return mesh, t, p_time_real, frequencies, P_freq

estimate_boundary_impedance(Lx, Ly, f_max, c, Z, source_center)

#mesh, t, p_time, frequencies, P_freq = frequency_to_time_domain(Lx, Ly, f_min, f_max, c, Z)
#anim = animate_pressure_field(mesh, t, p_time, save_file='wave_propagation.gif')