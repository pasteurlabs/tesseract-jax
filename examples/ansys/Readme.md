# ANSYS Tesseract Integration

This directory contains an example Tesseract configuration and scripts demonstrating how to use Tesseract-JAX with ANSYS spaceclaim and PyMAPDL. 

## Get Started

### SpaceClaim Tesseract

On a machine that has ANSYS SpaceClaim installed, create a new python environment and install the required dependencies by running:

```bash
pip install -r spaceclaim/tesseract_requirements.txt
```

Then navigate to the `spaceclaim` directory and start the Tesseract runtime server with:

```bash
uv run tesseract-runtime serve
```

Note that we dont build a Tesseract image for SpaceClaim in this example. This is because SpaceClaim cannot be installed in a containerized environment.

### PyMAPDL Tesseract

On 