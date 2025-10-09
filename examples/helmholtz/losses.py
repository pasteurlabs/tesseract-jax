import jax.numpy as jnp

def get_quadrature_point_coordinates(fe):
    """
    Get physical coordinates of all quadrature points.
    
    Returns:
        coords: [n_cells, n_quad_per_cell, dim]
    """
    n_cells = fe.num_cells
    n_quad = fe.num_quads
    dim = fe.dim
    
    coords = jnp.zeros((n_cells, n_quad, dim))
    
    for cell_idx in range(n_cells):
        cell_nodes = fe.cells[cell_idx]
        node_coords = fe.points[cell_nodes]  # [n_nodes_per_cell, dim]
        
        for quad_idx in range(n_quad):
            # Shape functions at this quadrature point
            shape_vals = fe.shape_vals[quad_idx]  # [n_nodes_per_cell]
            
            # Physical coordinate = Σ N_i * x_i
            phys_coord = jnp.sum(
                shape_vals[:, None] * node_coords,
                axis=0
            )
            coords = coords.at[cell_idx, quad_idx].set(phys_coord)
    
    return coords

def acoustic_focus_loss(problem, sol_list_pred, zone, quad_coords=None):
    """
    Compute loss in a zone around a receiver.
    
    Args:
        receiver_zone: dict with 'center', 'radius'
            Example: {'center': [0.3, 0.7], 'radius': 0.1}
    """
    p_pred_quad = problem.fes[0].convert_from_dof_to_quad(sol_list_pred[0])

    if quad_coords is None:
        quad_coords = get_quadrature_point_coordinates(problem.fes[0])
    JxW = problem.fes[0].JxW[:, :, None]
    
    center = jnp.array(zone['center'])
    radius = zone['radius']
    
    # Distance from zone center
    distances = jnp.linalg.norm(
        quad_coords - center[None, None, :],
        axis=2,
        keepdims=True
    )
    
    # Mask for points in this zone
    in_zone = distances < radius
    
    # Compute loss in this zone
    error_squared = jnp.abs(p_pred_quad)**2
    error_squared_zeroed = jnp.where(in_zone, error_squared * JxW, 0.0)
    print("Zone nonzero count:", jnp.count_nonzero(error_squared_zeroed))
    zone_loss = -jnp.sum(error_squared_zeroed)
    
    return zone_loss

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