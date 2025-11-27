# Grid fin optimization with ANSYS SpaceClaim and PyANSYS

We want to perform a parameter optimization on a SpaceX-style grid fin geometry to maximize stiffness while maintaining a strict mass limit. This ensures the fin remains rigid during re-entry Max-Q (maximum dynamic pressure) for robust aerodynamic control. To demonstrate a realistic engineering workflow, we demonstrate a gradient based end to optimization that plugs into ANSYS Spaceclaim as a design software and uses PyANSYS to setup a differentiable finite element simulation. The workflow involves three Tesseracts:


- **Ansys SpaceClaim Tesseract** takes a set of differentiable and non differentiable parameters and injects them into a SpaceClaim script that generates the grid fin geometry. It then returns returns the resulting triangular surface mesh as a list of points and faces. More about this here: ...

- **SDF & Finite Difference Tesseract** takes the same set of geometry parameters, passes them to the Spaceclaim Tesseract (Or any other Tesseract that returns a polygon mesh) and computes the signed distance field for the geometry on a regular grid. Additionally, the Tesseract is differentiable and computes Jacobian of the SDF field with respect to the design space parameters using finite differences. It has some additional features, such as precomputing the Jacobian while the rest of the pipeline is busy with computing primals and vector jacobian products (vjp) of later steps. In theory, we can plug any Tesseract that conforms to the in and output schema of the SpaceClaim Tesseract.

- **PyMapDL Tesseract** takes a hex mesh, the boundary descriptions and computes the strain energy for all cells as well as the total compliance. It uses the PyMapDL library to setup a fully differentiable FEM solver for linear elasticity. More about this here: ...

The Tesseracts are then composed into a fully differentiable workflow using Tesseract-JAX.

![Workflow](workflow_1.png)

In order to der


Note that the function that computes the boundary conditions and constructs the hex mesh do NOT need to be differentiable, as we are only differentiating with respect to the quantities that are carried on mesh cells, but not the mesh itself.
