from dataclasses import dataclass
import jax.numpy as jnp
import numpy as np
from jax_fem.problem import Problem


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

class AcousticHelmholtzImpedance(Problem):
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


class AcousticHelmholtzNeumann(Problem):
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
        """Neumann condition: ∂p/∂n = 0"""
            
        def surface_map(u, x):
            return 0 * u

        return [surface_map] * len(self.internal_vars_surfaces)