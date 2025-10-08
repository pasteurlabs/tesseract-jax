import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def animate_pressure_field(mesh, t, p_time, save_file=None):
    """
    Animate pressure field evolution in time.
    
    Args:
        mesh: JAX-FEM mesh
        t: Time array
        p_time: Pressure field [n_nodes, n_time]
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Initial plot
    vmax = jnp.max(jnp.abs(p_time))
    tri_plot = ax.tricontourf(
        mesh.points[:, 0],
        mesh.points[:, 1],
        p_time[:, 0],
        levels=50,
        cmap='RdBu_r',
        vmin=-vmax,
        vmax=vmax
    )
    
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_aspect('equal')
    title = ax.set_title(f'Time: {t[0]*1000:.2f} ms')
    cbar = plt.colorbar(tri_plot, ax=ax, label='Pressure (Pa)')
    
    def update(frame):
        ax.clear()
        tri_plot = ax.tricontourf(
            mesh.points[:, 0],
            mesh.points[:, 1],
            p_time[:, frame],
            levels=50,
            cmap='RdBu_r',
            vmin=-vmax,
            vmax=vmax
        )
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_aspect('equal')
        ax.set_title(f'Time: {t[frame]*1000:.2f} ms')
        return tri_plot,
    
    # Animate (use every Nth frame for speed)
    skip = max(1, len(t) // 200)  # Max 200 frames
    anim = FuncAnimation(
        fig, 
        update, 
        frames=range(0, len(t), skip),
        interval=50,  # 50 ms between frames
        blit=False
    )
    
    if save_file:
        anim.save(save_file, writer='pillow', fps=20)
    
    plt.show()
    return anim

def plot_pressure_field(sol, mesh, k, c):
    """Plot pressure magnitude and phase at single frequency"""
    f = k * c / (2 * jnp.pi)
    
    # Reshape solution if needed
    pressure = sol[:, 0]  # Extract pressure values
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Magnitude
    scatter1 = ax1.tricontourf(
        mesh.points[:, 0], 
        mesh.points[:, 1], 
        jnp.abs(pressure),
        levels=50,
        cmap='viridis'
    )
    ax1.set_title(f'Pressure Magnitude at {f:.0f} Hz')
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.set_aspect('equal')
    plt.colorbar(scatter1, ax=ax1, label='|p| (Pa)')
    
    # Phase
    scatter2 = ax2.tricontourf(
        mesh.points[:, 0], 
        mesh.points[:, 1], 
        jnp.angle(pressure) * 180/jnp.pi,
        levels=50,
        cmap='twilight'
    )
    ax2.set_title(f'Pressure Phase at {f:.0f} Hz')
    ax2.set_xlabel('x (m)')
    ax2.set_ylabel('y (m)')
    ax2.set_aspect('equal')
    plt.colorbar(scatter2, ax=ax2, label='∠p (degrees)')
    
    plt.tight_layout()
    plt.show()