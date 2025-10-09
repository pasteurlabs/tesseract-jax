import logging
from jax_fem import logger

from experiments import estimate_boundary_impedance, estimate_impedance_max_focus, frequency_to_time_domain, frequency_to_time_domain_circular
from visualization import animate_pressure_field

logger.setLevel(logging.NOTSET)

# Problem parameters
f_min = 100 # Hz
f_max = 1000 # Hz
c = 343 # speed of sound m/s
beta = 1.5 + 0.3j # admittance


# source_center = [1.0, 1.0] # REMEMBER: is using quads, not centered at 0.0
# estimate_boundary_impedance(Lx, Ly, f_max, c, beta, source_center)


Lx = 3.0
Ly = 3.0
receiver_zone = {
    "center": [1.5, 0.0],
    "radius": 0.1
}
source_center = [-1.5, 1.0]
estimate_impedance_max_focus(Lx, Ly, f_max, c, beta, source_center, receiver_zone)

# source_center = [0.0, 0.0] # REMEMBER: is using tri, centered at 0.0
# mesh, t, p_time, frequencies, P_freq = frequency_to_time_domain(Lx, Ly, f_min, f_max, c, beta, source_center)
# anim = animate_pressure_field(mesh, t, p_time, save_file='wave_propagation.gif')

radius = 1
# source_center = [0.0, 0.0] # REMEMBER: is using tri, centered at 0.0
# mesh, t, p_time, frequencies, P_freq = frequency_to_time_domain_circular(radius, f_min, f_max, c, beta, source_center)
# anim = animate_pressure_field(mesh, t, p_time, save_file='wave_propagation.gif')
