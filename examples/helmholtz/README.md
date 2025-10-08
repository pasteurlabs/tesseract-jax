### Installation

#### Jax-FEM installations

If you want a clean install, create a new virtual environment
```bash
uv venv
source .venv/bin/activate
```

```bash
uv pip install "https://github.com/deepmodeling/jax-fem.git"
uv pip install pyfiglet meshio gmsh optax fenics-basix matplotlib
brew install petsc-complex==3.24.0
uv pip install petsc==3.24.0
```

Set the correct path

```bash
export PETSC_DIR=$(brew --prefix petsc-complex)
export PETSC_ARCH=""

uv cache clean
uv pip install petsc4py==3.24.0 --no-cache-dir -v
```

Should then show:

```bash
DEBUG PETSC_DIR:    /opt/homebrew/opt/petsc-complex
DEBUG PETSC_ARCH:   
DEBUG version:      3.24.0 release
DEBUG integer-size: 32-bit
DEBUG scalar-type:  complex
DEBUG precision:    double
DEBUG language:     CONLY
DEBUG compiler:     mpicc
DEBUG linker:       mpicc
```

#### PyTorch3D installation

```bash
uv pip install --upgrade jupyter ipywidgets
uv pip install torch plotly nbformat
uv pip install "git+https://github.com/facebookresearch/pytorch3d.git" --no-build-isolation
```