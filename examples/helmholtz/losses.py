import jax.numpy as jnp

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

def J_objective(Z_params, fwd_pred, problem, p_ref):
    p_pred = fwd_pred(Z_params)
    data_loss = compute_acoustic_loss(
        problem, p_pred, p_ref,
        w_mag=0.5,
        w_phase=0.5,
        w_rel=0.0 # Helps with scaling
        )

    return data_loss