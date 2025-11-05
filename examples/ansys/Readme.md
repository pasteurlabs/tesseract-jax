# ANSYS Tesseract Integration

This directory contains an example Tesseract configuration and scripts demonstrating how to use Tesseract-JAX with ANSYS spaceclaim and PyMAPDL.

## Get Started

### SpaceClaim Tesseract

On a windows machine, install [ansys-spaceclaim](https://www.ansys.com/products/3d-design/ansys-spaceclaim) and create a new python env. Assuming you using windows powerhsell, install the required dependencies:

```bash
pip install tesseract-core[runtime]
```

Clone this repository, navigate to the `examples/ansys/spaceclaim` directory and start the Tesseract runtime server with:

```bash
tesseract-runtime serve
```
Note that we dont build a Tesseract image for SpaceClaim in this example. This is because SpaceClaim cannot be installed in a containerized environment. You can test it using:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/apply" -Method Post -Body (
    @{
        inputs = @{
            grid_parameters = @{
                "__params__.z2"  = "200"
                "__params__.z3"  = "600"
                "__params__.s1"  = "0"
                "__params__.s2"  = "1 * (math.pi / 8)"
                "__params__.s3"  = "2 * (math.pi / 8)"
                "__params__.s4"  = "3 * (math.pi / 8)"
                "__params__.s5"  = "4 * (math.pi / 8)"
                "__params__.s6"  = "5 * (math.pi / 8)"
                "__params__.s7"  = "6 * (math.pi / 8)"
                "__params__.s8"  = "7 * (math.pi / 8)"
                "__params__.e1"  = "(0) + math.pi"
                "__params__.e2"  = "(1 * (math.pi / 8)) + math.pi"
                "__params__.e3"  = "(2 * (math.pi / 8)) + math.pi"
                "__params__.e4"  = "(3 * (math.pi / 8)) + math.pi"
                "__params__.e5"  = "(4 * (math.pi / 8)) + math.pi"
                "__params__.e6"  = "(5 * (math.pi / 8)) + math.pi"
                "__params__.e7"  = "(6 * (math.pi / 8)) + math.pi"
                "__params__.e8"  = "(7 * (math.pi / 8)) + math.pi"
            }
        }
    } | ConvertTo-Json -Depth 10
) -ContentType "application/json"
```

### PyMAPDL Tesseract

On
