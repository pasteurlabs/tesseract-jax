import jax
import jax.numpy as jnp
import numpy as np
import optax
import matplotlib.pyplot as plt
from jax_fem.solver import ad_wrapper, solver

from mesh_setup import create_circular_mesh, create_rectangular_mesh
from problems import AcousticHelmholtzImpedance, AcousticHelmholtzNeumann, Source
from losses import J_objective, compute_acoustic_loss

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
        loss_fn = lambda Z: J_objective(Z, fwd_pred, problem, measurements)
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
    mesh, location_fns2, ele_type = create_rectangular_mesh(Lx, Ly, c, f_max, ppw)

    k_max = 2 * jnp.pi * f_max / c # wave number    
    source = Source(k_max=k_max, center=source_center, amplitude=1000)

    # Create problem instance
    problem = AcousticHelmholtzImpedance(
        mesh=mesh, k=k_max, source_params=source,
        vec=1, dim=2, ele_type=ele_type,
        # dirichlet_bc_info=dirichlet_bc_info, 
        location_fns=location_fns2,
        gauss_order=1)    
    
    fwd_pred = ad_wrapper(problem)

    measurements = fwd_pred(Z_true)

    # Add noise to measurements
    # measurements = measurements + 0.01 * jax.random.normal(jax.random.PRNGKey(0), measurements.shape)

    # test_impedance_sensitivity(problem, fwd_pred, Z_true=1.5+0.3j)
    
    # Optimize
    Z_opt, losses, Z_opt_list = optimize_impedance(fwd_pred, problem, measurements, n_iterations=n_iterations, learning_rate=learning_rate)

    print(f"\nTrue impedance: {Z_true}")
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

    mesh, location_fns2, ele_type = create_rectangular_mesh(Lx, Ly, c, f_max, ppw)

    N_freq = int((f_max - f_min) / df) + 1
    frequencies = np.linspace(f_min, f_max, N_freq)
            
    source = Source(k_max=k_max, center=source_center, amplitude=1000)

    pressure_freq = []

    for i, f in enumerate(frequencies):
        k = 2 * jnp.pi * f / c
        
        # Create problem instance
        problem = AcousticHelmholtzImpedance(
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

def frequency_to_time_domain_circular(
    radius,
    f_min,
    f_max,
    c,
    Z,    
    source_center=[1.0, 1.0],
    ppw = 2.0,
):
    df = 10 # Hz (frequency resolution) TODO: calculate from time requirement?
    k_max = 2 * jnp.pi * f_max / c # wave number

    mesh, location_fns2, ele_type = create_circular_mesh(radius, c, f_max, ppw)

    N_freq = int((f_max - f_min) / df) + 1
    frequencies = np.linspace(f_min, f_max, N_freq)
            
    source = Source(k_max=k_max, center=source_center, amplitude=1000)

    pressure_freq = []

    for i, f in enumerate(frequencies):
        k = 2 * jnp.pi * f / c
        
        # Create problem instance
        problem = AcousticHelmholtzNeumann(
            mesh=mesh, k=k, source_params=source,
            vec=1, dim=2, ele_type=ele_type,
            # dirichlet_bc_info=dirichlet_bc_info, 
            location_fns=location_fns2,
            gauss_order=1)

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

def test_impedance_sensitivity(problem, fwd_pred, Z_true, w_mag=1.0, w_phase=1.0, w_rel=0.0):
    """
    Test how sensitive the pressure field is to real vs imaginary Z.
    """    
    
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