import jax
import jax.numpy as jnp
import numpy as np
import optax
import matplotlib.pyplot as plt
from jax_fem.solver import ad_wrapper, solver

from mesh_setup import create_circular_mesh, create_rectangular_mesh_quads, create_rectangular_mesh_triangular
from problems import AcousticHelmholtzImpedance, Source
from losses import acoustic_focus_loss, compute_acoustic_loss, compute_acoustic_loss_zone, get_quadrature_point_coordinates

plt.matplotlib.rcParams.update({'font.size': 22})
linewidth=4

def optimize_impedance_focus_point(fwd_pred, problem, receiver_zone, beta, n_iterations, learning_rate):
    """Optimize impedance to match measurements."""
    
    # Set up optimizer
    optimizer = optax.sgd(learning_rate)
    opt_state = optimizer.init(beta)

    beta_opt = beta
    
    losses = []
    beta_opt_list = []
    
    # should be static!
    quad_coords = get_quadrature_point_coordinates(problem.fes[0])
    
    for i in range(n_iterations):
        # Compute loss and gradient
        loss_fn = lambda beta: acoustic_focus_loss(
            problem, 
            fwd_pred(beta), 
            receiver_zone,
            quad_coords,
        )
        loss, grad = jax.value_and_grad(loss_fn)(beta_opt)
        grad = jnp.real(grad) -1j * jnp.imag(grad)
        
        # Update parameters
        updates, opt_state = optimizer.update(grad, opt_state)
        beta_opt = optax.apply_updates(beta_opt, updates)
        
        losses.append(loss)
        beta_opt_list.append(beta_opt)
        
        # Warning signs
        grad_norm = jnp.linalg.norm(grad)
        if grad_norm > 1e3:
            print("⚠️ WARNING: Gradient explosion!")
        if grad_norm < 1e-8:
            print("⚠️ WARNING: Gradient vanishing!")
            
        if i % 10 == 0:
            print(f"\n*** Iteration {i}: Loss = {loss:.6f}, Z = {beta_opt} ***\n")
    
    return beta_opt, losses, beta_opt_list

def optimize_impedance(fwd_pred, problem, measurements, receiver_zones, n_iterations, learning_rate):
    """Optimize impedance to match measurements."""    

    # Initialize impedance parameters
    # Could be scalar, or array for spatially-varying impedance    
    beta_init = 3.0 + 0.05j
    
    # Set up optimizer
    optimizer = optax.sgd(learning_rate)
    opt_state = optimizer.init(beta_init)

    beta_opt = beta_init
    
    losses = []
    beta_opt_list = []
    
    # should be static!
    quad_coords = get_quadrature_point_coordinates(problem.fes[0])

    for i in range(n_iterations):
        # Compute loss and gradient
        loss_fn = lambda beta: compute_acoustic_loss_zone(
            problem, fwd_pred(beta)[0], measurements, receiver_zones, quad_coords,
            w_mag=0.5,
            w_phase=0.1,
            w_rel=5.0 # Helps with scaling
        )
        loss, grad = jax.value_and_grad(loss_fn)(beta_opt)
        grad = jnp.real(grad) -1j * jnp.imag(grad)
        
        # Update parameters
        updates, opt_state = optimizer.update(grad, opt_state)
        beta_opt = optax.apply_updates(beta_opt, updates)
        
        losses.append(loss)
        beta_opt_list.append(beta_opt)
        
        # Warning signs
        grad_norm = jnp.linalg.norm(grad)
        if grad_norm > 1e3:
            print("⚠️ WARNING: Gradient explosion!")
        if grad_norm < 1e-8:
            print("⚠️ WARNING: Gradient vanishing!")
            
        if i % 10 == 0:
            print(f"\n*** Iteration {i}: Loss = {loss:.6f}, Z = {beta_opt} ***\n")
    
    return beta_opt, losses, beta_opt_list

def estimate_boundary_impedance(
    Lx, Ly,
    f_max,
    c,
    beta_true,
    source_center,
    receiver_zones,
    ppw = 5.0,
    n_iterations = 200,
    learning_rate = 0.1,
    noise_level = 0.02,
    save_plots=False,
): 
    mesh, location_fns, ele_type = create_rectangular_mesh_quads(Lx, Ly, c, f_max, ppw, uniform_bc=False)
    # NOTE: Gradients are vanishing for triangular mesh
    #mesh, location_fns, ele_type = create_rectangular_mesh_triangular(Lx, Ly, c, f_max, ppw, uniform_bc=False)

    k_max = 2 * jnp.pi * f_max / c # wave number    
    source = Source(k_max=k_max, center=source_center, amplitude=1000)

    # Create problem instance
    problem = AcousticHelmholtzImpedance(
        mesh=mesh, k=k_max, source_params=source,
        vec=1, dim=2, ele_type=ele_type,
        location_fns=location_fns,
        gauss_order=1)    
    
    fwd_pred = ad_wrapper(problem)
    measurements = fwd_pred(beta_true)

    # Add noise to measurements
    measurements = measurements[0] + noise_level * jax.random.normal(jax.random.PRNGKey(0), measurements[0].shape)
    
    # Optimize
    Z_opt, losses, Z_opt_list = optimize_impedance(fwd_pred, problem, measurements, receiver_zones, n_iterations=n_iterations, learning_rate=learning_rate)

    print(f"\nTrue admittance: {beta_true}")
    print(f"Estimated admittance: {Z_opt}")
    
    if save_plots:
        # Plot convergence
        plt.figure(figsize=(10, 8))
        plt.semilogy(losses, linewidth=4)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.grid()
        plt.savefig(f"convergence.png", bbox_inches='tight', pad_inches=0)
        
        plt.figure(figsize=(10, 8))
        plt.plot(np.real(Z_opt_list), linewidth=4)
        plt.xlabel('Iteration')
        plt.ylabel(r'$\beta$ (real)')
        plt.grid()
        plt.savefig(f"betal_real.png", bbox_inches='tight', pad_inches=0)
        
        plt.figure(figsize=(10, 8))
        plt.plot(np.imag(Z_opt_list), linewidth=4)
        plt.xlabel('Iteration')
        plt.ylabel(r'$\beta$ (imag)')
        plt.grid()
        plt.savefig(f"betal_imag.png", bbox_inches='tight', pad_inches=0)
    else:
        # Plot convergence
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 3, 1)
        plt.semilogy(losses, linewidth=4)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.grid()        
        plt.title('Convergence')
        
        plt.subplot(1, 3, 2)
        plt.plot(np.real(Z_opt_list), linewidth=4)
        plt.xlabel('Iteration')
        plt.ylabel(r'$\beta$ (real)')
        plt.grid()
        plt.title('Normalized admittance (real)')
        
        plt.subplot(1, 3, 3)
        plt.plot(np.imag(Z_opt_list), linewidth=4)
        plt.xlabel('Iteration')
        plt.ylabel(r'$\beta$ (imag)')
        plt.grid()
        plt.title('Normalized admittance (imag)')
        plt.show()

def frequency_to_time_domain(
    Lx, Ly,
    f_min,
    f_max,
    c,
    beta,
    source_center,
    ppw = 5.0,
):
    df = 10 # Hz (frequency resolution) TODO: calculate from time requirement?
    k_max = 2 * jnp.pi * f_max / c # wave number

    
    mesh, location_fns, ele_type = create_rectangular_mesh_triangular(Lx, Ly, c, f_max, ppw, uniform_bc=True)
    # mesh, location_fns2, ele_type = create_rectangular_mesh_quads(Lx, Ly, c, f_max, ppw)

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
            location_fns=location_fns,
            gauss_order=1)        
        problem.set_params(beta)

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
    beta,
    source_center,
    ppw = 5.0,
):
    df = 10 # Hz (frequency resolution) TODO: calculate from time requirement?
    k_max = 2 * jnp.pi * f_max / c # wave number

    mesh, location_fns, ele_type = create_circular_mesh(radius, c, f_max, ppw)

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
            location_fns=location_fns,
            gauss_order=1)        
        problem.set_params(beta)

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