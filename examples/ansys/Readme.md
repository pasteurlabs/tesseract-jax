# ANSYS Tesseract Integration

This directory contains an example Tesseract configuration and scripts demonstrating how to use Tesseract-JAX with ANSYS spaceclaim and PyMAPDL.

## Get Started

### PL internal instructions:

- Our open ports are: 443 and 50052.
- Make sure to be connected to the PL VPN.

### Prerequisites

For the windows machine:
1. ANSYS installed and an active license.
2. Python and a python environment (e.g., conda, venv).
3. Two open ports.

For the linux machine:
1. Docker installed and running.
2. Python and a python environment (e.g., conda, venv).

### SpaceClaim Tesseract

Create a new python env. Assuming you using windows powerhsell, install the required dependencies:

```bash
pip install tesseract-core[runtime]
```

Clone this repository, navigate to the `examples/ansys/spaceclaim_tess` directory and start the Tesseract runtime server with:

```bash
tesseract-runtime serve --port <port_number_1>
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

### PyMAPDL Server

On a windows machine, make sure ansys is installed. Then run the following powershell command to start ansys with grpc server enabled:

```powershell
Start-Process -FilePath "F:\ANSYS Inc\v242\ansys\bin\winx64\ANSYS242.exe" -ArgumentList "-grpc", "-port", "<port_number_2>"
```

replace "v242" with your ansys version.

### Build tesseracts

1. Obtain the ip adress of the windows machine by running:

```powershell
(Get-NetIPAddress -AddressFamily IPv4 -InterfaceAlias "Wi-Fi","Ethernet" | Where-Object {$_.IPAddress -notlike "169.254.*" -and $_.IPAddress -ne $null}).IPAddress
```
2. On the linux machine, create a new python env and install tesseract-core with:

```bash
pip install tesseract-core[runtime]
```

3. Build all relevant tesseracts:

```bash
tesseract build fem_tess
tesseract build pymapdl_tess
tesseract build meshing_tess
```
