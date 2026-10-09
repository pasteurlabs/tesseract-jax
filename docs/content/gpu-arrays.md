# GPU arrays

When a call is compiled for a CUDA device, `apply_tesseract` keeps GPU arrays on the device whenever it can, and copies them through the host otherwise. Either way the results are the same, so you do not need to change any code; only the speed differs. GPU transports are an experimental tesseract-core feature.

## Serving a Tesseract with `cuda_ipc`

To exchange GPU arrays with a served Tesseract without a host copy, serve it with the `cuda_ipc` transport. This tells the Tesseract's runtime that its endpoints accept GPU arrays as inputs, so only enable it for Tesseracts written to handle them:

```python
import jax
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract

with Tesseract.from_source("tesseract_api.py", gpu_transport="cuda_ipc") as tess:
    out = jax.jit(lambda x: apply_tesseract(tess, {"x": x}))(x)
```

`Tesseract.from_image(..., gpus=["all"], gpu_transport="cuda_ipc")` works the same way for a containerized Tesseract.

## How the transport is picked

Before the first call compiled for a CUDA device, tesseract-jax asks the Tesseract which transport to use (via `Tesseract.resolve_gpu_transport`). That checks once per connection that the transport actually works between your process and the server, by exchanging a small GPU array:

- A Tesseract created with `gpu_transport="cuda_ipc"` uses it. If it does not work from your process, the call raises with the reason.
- A Tesseract that requests no transport, such as one from `Tesseract.from_url`, uses `cuda_ipc` if the server offers it and it works. Otherwise its GPU arrays are copied through the host, with a one-time warning saying why. This happens, for example, when the server runs on another machine.
- Calls compiled for CPU always go through the host and never run the check.

To always copy GPU arrays through the host, pass a view of the Tesseract:

```python
apply_tesseract(tess.with_encoding(gpu_transport="none"), inputs)
```

## In-process Tesseracts

A Tesseract loaded with `Tesseract.from_tesseract_api` receives host (NumPy) arrays by default. Create it with `gpu_transport="cuda_ipc"` to hand its endpoints the device buffers as they are, as objects exposing `__cuda_array_interface__` that are only valid during the call:

```python
tess = Tesseract.from_tesseract_api("tesseract_api.py", gpu_transport="cuda_ipc")
```

Its endpoints may return host or device arrays either way.

The endpoints must not modify these input arrays in place. They are JAX's own buffers, which hold the caller's arrays and may be read again by later operations, so an in-place change such as `x -= x.mean()` silently changes the caller's values. Compute into a new array instead (`x = x - x.mean()`). A served Tesseract is not affected, since it receives its own copy of each input.
