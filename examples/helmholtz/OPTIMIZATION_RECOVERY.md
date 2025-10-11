# Optimization Recovery Guide

## Overview

The `mesh_acoustic_optimization.ipynb` notebook now includes robust error handling and incremental saving to handle solver failures gracefully. If the optimization fails at any point, all progress up to the failure is preserved and can be visualized.

## How It Works

### Incremental Saving

After **every iteration**, the notebook saves:
- Complete optimization history to `optimization_outputs/history.pkl`
- Visualization PNG files every N iterations (default: every 5 iterations)

This means if the solver fails at iteration 25 of 50, you still have:
- All data from iterations 0-24
- Visualizations at iterations 0, 5, 10, 15, 20

### Error Handling

The optimization function includes multiple layers of error handling:

1. **Per-iteration try-catch**: Each iteration wrapped in try-catch to catch solver failures
2. **Keyboard interrupt**: Ctrl+C cleanly saves and exits
3. **Finally block**: Always saves final state, even on unexpected errors

### What Gets Saved

The `history.pkl` file contains:

```python
{
    'acoustic_energy': [],      # Energy at each iteration
    'mesh_loss': [],            # Mesh quality loss
    'area': [],                 # Room area (should stay constant)
    'mesh_history': [],         # Mesh vertex positions at each iteration
    'pressure_fields': [],      # Pressure fields (saved at visualization steps)
    'completed_iterations': 0   # Number of successful iterations
}
```

## Recovering from Failures

If the optimization fails mid-run, you can load and visualize the partial results:

```python
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load saved history
with open('optimization_outputs/history.pkl', 'rb') as f:
    history = pickle.load(f)

print(f"Completed {history['completed_iterations']} iterations")

# Plot energy reduction
plt.figure(figsize=(10, 6))
plt.plot(history['acoustic_energy'])
plt.xlabel('Iteration')
plt.ylabel('Acoustic Energy')
plt.title('Energy Reduction (Partial Run)')
plt.yscale('log')
plt.grid(True)
plt.show()

# Visualize final mesh before failure
final_mesh = history['mesh_history'][-1]
plt.figure(figsize=(8, 8))
plt.triplot(final_mesh[:, 0], final_mesh[:, 1], cells, 'k-', linewidth=0.5)
plt.axis('equal')
plt.title(f'Mesh at Iteration {history["completed_iterations"]}')
plt.show()

# Check area constraint
plt.figure(figsize=(10, 6))
plt.plot(history['area'])
plt.axhline(y=target_area, color='r', linestyle='--', label='Target Area')
plt.xlabel('Iteration')
plt.ylabel('Area (m²)')
plt.title('Area Constraint History')
plt.legend()
plt.grid(True)
plt.show()
```

## Creating Animations from Partial Results

Even if optimization didn't complete, you can create animations from the saved images:

```python
import glob
from PIL import Image

# Load all saved images (even from partial run)
image_files = sorted(glob.glob('optimization_outputs/iter_*.png'))
print(f"Found {len(image_files)} saved frames")

if len(image_files) > 0:
    images = [Image.open(f) for f in image_files]

    # Create GIF
    images[0].save(
        'optimization_outputs/partial_optimization.gif',
        save_all=True,
        append_images=images[1:],
        duration=500,
        loop=0
    )
    print("✓ Animation created from partial results")
```

## Tips for Handling Solver Failures

### 1. Adjust Solver Tolerance

If solver fails frequently, try relaxing tolerance in Cell 6:

```python
solver_options = {
    'petsc_solver': {
        'ksp_type': 'gmres',
        'ksp_gmres_restart': 100,
        'pc_type': 'ilu',
        'pc_factor_levels': 3,
        'pc_factor_fill': 2.0,
        'ksp_rtol': 1e-4,      # Relaxed from 1e-5
        'ksp_max_it': 3000,    # Increased from 2000
    },
    'tol': 1e-3                # Relaxed from 1e-4
}
```

### 2. Reduce Learning Rate

Aggressive deformations can create poorly-conditioned meshes. Try smaller learning rates:

```python
optimized_mesh, history = optimize_room_shape(
    initial_points, cells, location_fns,
    n_iterations=50,
    lr_boundary=0.002,   # Reduced from 0.005
    lr_internal=0.03,    # Reduced from 0.05
    ...
)
```

### 3. Increase Mesh Regularization

Stronger mesh quality enforcement prevents extreme deformations:

```python
# In Cell 8, modify compute_mesh_regularization_torch:
total_loss = 1.0 * loss_edge + 2.0 * loss_laplacian + 1.0 * loss_normal
#            ^^^                ^^^                    ^^^
#         increased weights
```

### 4. Save More Frequently

To minimize data loss, save visualizations more often:

```python
optimized_mesh, history = optimize_room_shape(
    ...,
    save_every=2  # Save every 2 iterations instead of 5
)
```

## Understanding Failure Modes

### Solver Convergence Failure
**Symptom**: "Linear solver did not converge" or "Maximum iterations reached"
**Cause**: Mesh deformation created poorly-conditioned system
**Solution**: Reduce learning rates, increase mesh regularization

### Mesh Quality Degradation
**Symptom**: Inverted elements, extreme aspect ratios
**Cause**: Boundary moves too aggressively, internal points can't keep up
**Solution**: Increase `internal_steps` (more internal optimization), reduce `lr_boundary`

### Area Constraint Violation
**Symptom**: Area drifts away from target
**Cause**: Area weight too low relative to other objectives
**Solution**: Increase `area_weight` parameter

## File Locations

All outputs are saved to `optimization_outputs/`:
- `history.pkl` - Complete optimization history (updated every iteration)
- `iter_000.png`, `iter_005.png`, ... - Visualization snapshots
- `optimization_animation.gif` - Animated visualization (created at end)

## Example: Resume from Checkpoint

If you want to continue optimization from where it left off:

```python
import pickle

# Load previous history
with open('optimization_outputs/history.pkl', 'rb') as f:
    previous_history = pickle.load(f)

print(f"Previous run completed {previous_history['completed_iterations']} iterations")

# Get final mesh from previous run
last_mesh = previous_history['mesh_history'][-1]

# Continue optimization from this mesh
optimized_mesh, new_history = optimize_room_shape(
    last_mesh,  # Start from last state
    cells, location_fns,
    n_iterations=25,  # Continue for 25 more iterations
    ...
)

# Merge histories
combined_history = {
    'acoustic_energy': previous_history['acoustic_energy'] + new_history['acoustic_energy'],
    'mesh_loss': previous_history['mesh_loss'] + new_history['mesh_loss'],
    'area': previous_history['area'] + new_history['area'],
    'mesh_history': previous_history['mesh_history'] + new_history['mesh_history'],
    'completed_iterations': previous_history['completed_iterations'] + new_history['completed_iterations']
}
```

## Questions?

If you encounter issues:
1. Check the console output for error messages
2. Load `history.pkl` to inspect progress
3. Visualize intermediate meshes to identify when problems started
4. Adjust solver/optimization parameters as needed
5. Try starting from a different initial mesh or source location

The robust error handling ensures you never lose work, even if the solver becomes unstable during optimization!
