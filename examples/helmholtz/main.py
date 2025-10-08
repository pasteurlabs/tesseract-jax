import logging
from jax_fem import logger

from experiments import estimate_boundary_impedance, frequency_to_time_domain, frequency_to_time_domain_circular
from visualization import animate_pressure_field

logger.setLevel(logging.NOTSET)

# Problem parameters
radius = 1
Lx = 2.0
Ly = 2.0
f_min = 100 # Hz
f_max = 1000 # Hz
c = 343 # speed of sound m/s
Z = 1.5 + 0.3j # impedance

source_center = [1.0, 1.0]

# estimate_boundary_impedance(Lx, Ly, f_max, c, Z, source_center)

# mesh, t, p_time, frequencies, P_freq = frequency_to_time_domain(Lx, Ly, f_min, f_max, c, Z)
# anim = animate_pressure_field(mesh, t, p_time, save_file='wave_propagation.gif')
mesh, t, p_time, frequencies, P_freq = frequency_to_time_domain_circular(radius, f_min, f_max, c)
# anim = animate_pressure_field(mesh, t, p_time, save_file='wave_propagation.gif')
