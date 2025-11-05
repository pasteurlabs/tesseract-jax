# ANSYS Tesseract Integration

This directory contains an example Tesseract configuration and scripts demonstrating how to use Tesseract-JAX with ANSYS spaceclaim and PyMAPDL.

## Get Started

### SpaceClaim Tesseract

On a windows machine, install [ansys-spaceclaim](https://www.ansys.com/products/3d-design/ansys-spaceclaim) and create a new python env. Install the required dependencies:

```bash
pip install tesseract-core[runtime]
```

Clone this repository, navigate to the `examples/ansys/spaceclaim` directory and start the Tesseract runtime server with:

```bash
tesseract-runtime serve
```

Note that we dont build a Tesseract image for SpaceClaim in this example. This is because SpaceClaim cannot be installed in a containerized environment.

### PyMAPDL Tesseract

On
