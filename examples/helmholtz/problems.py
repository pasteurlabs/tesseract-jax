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
        self.k = k
        self.source_params = source_params
        self.n_impedance_bcs = len(kargs["location_fns"])
        super().__init__(mesh, **kargs)

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
        import jax

        def surface_map(u, x, beta):
            # u here represents test/trial functions on the boundary element
            # This should return the integrand ik*beta * p * v
            # where beta is the normalized admittance rho*c*Z
            # and p and v are represented by u and 
            
            return 1j * self.k * beta * u

        # return [surface_map, surface_map, surface_map, surface_map]
        return [surface_map] * self.n_impedance_bcs


    def set_params(self, beta):
        """For inverse problems, the `set_params(params)` method provides the interface for dynamic model parameter updates."""
        
        Z_params_list = []
        # Get the quadrature points for each boundary surface
        for i in range(self.n_impedance_bcs):
            quad_surface_points = self.fes[0].get_physical_surface_quad_points(self.boundary_inds_list[i])
            Z_params = beta * np.ones_like(quad_surface_points[:, :, 0])
            Z_params_list.append([Z_params])

        self.internal_vars_surfaces = Z_params_list