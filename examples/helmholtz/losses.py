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

def compute_acoustic_loss_zone(problem, sol_pred, sol_ref, zones, quad_coords, w_mag, w_phase, w_rel):
    """
    Combined loss for acoustic inverse problems at multiple receiver zones.
    
    Args:
        problem: JAX-FEM problem object
        sol_pred: Predicted solution
        sol_ref: Reference solution
        zones: List of dicts, each with 'center' and 'radius'
               e.g., [{'center': [0.3, 0.7], 'radius': 0.1}, 
                      {'center': [0.8, 0.3], 'radius': 0.15}]
        w_mag: Weight for magnitude error
        w_phase: Weight for phase error
        w_rel: Weight for relative error term
        quad_coords: Quadrature point coordinates [n_cells, n_quad_per_cell, dim]
    
    Returns:
        total_loss: Combined loss only at receiver zones
    """
    
    # Convert solutions to quadrature points
    p_pred_quad = problem.fes[0].convert_from_dof_to_quad(sol_pred)
    p_ref_quad = problem.fes[0].convert_from_dof_to_quad(sol_ref)
    JxW = problem.fes[0].JxW[:, :, None]
    
    # Create combined mask for all receiver zones
    combined_mask = jnp.zeros((quad_coords.shape[0], quad_coords.shape[1], 1), dtype=bool)
    
    for zone in zones:
        center = jnp.array(zone['center'])
        radius = zone['radius']
        
        # Distance from this zone center
        distances = jnp.linalg.norm(
            quad_coords - center[None, None, :],
            axis=2,
            keepdims=True
        )
        
        # Mask for points in this zone
        in_zone = distances < radius
        
        # Add to combined mask (logical OR)
        combined_mask = combined_mask | in_zone
    
    # 1. Magnitude error (only in zones)
    mag_pred = jnp.abs(p_pred_quad)
    mag_ref = jnp.abs(p_ref_quad)
    mag_error_all = (mag_pred - mag_ref)**2 * JxW
    mag_error = jnp.sum(jnp.where(combined_mask, mag_error_all, 0.0))
    
    print("Zone nonzero count:", jnp.count_nonzero(jnp.where(combined_mask, mag_error_all, 0.0)))

    # 2. Phase error (only in zones)
    phase_diff = jnp.angle(p_pred_quad * jnp.conj(p_ref_quad))
    phase_error_all = phase_diff**2 * JxW
    phase_error = jnp.sum(jnp.where(combined_mask, phase_error_all, 0.0))
    
    # 3. Relative error (only in zones)
    relative_error_all = jnp.abs(p_pred_quad - p_ref_quad)**2 / (jnp.abs(p_ref_quad)**2 + 1e-8)
    relative_error_weighted = relative_error_all * JxW
    rel_loss = jnp.sum(jnp.where(combined_mask, relative_error_weighted, 0.0))
    
    # Combined loss
    total_loss = w_mag * mag_error + w_phase * phase_error + w_rel * rel_loss

    
    
    return total_loss

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