# GPU arrays

When a call is compiled for a CUDA device, `apply_tesseract` keeps GPU arrays on the device whenever it can, and copies them through the host otherwise. Either way the results are the same, so no code changes are needed. Only the speed differs. GPU transports are an experimental tesseract-core feature.

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

When a call is compiled for a CUDA device, tesseract-jax asks the Tesseract which transport to use (via `Tesseract.resolve_gpu_transport`). The Tesseract checks once per connection that the transport works between your process and the server, by exchanging a small GPU array:

- A Tesseract created with `gpu_transport="cuda_ipc"` uses `cuda_ipc`. If `cuda_ipc` does not work from your process, for example because the server does not share your GPU, the call raises with the reason.
- A Tesseract that requests no transport, such as one from `Tesseract.from_url`, uses `cuda_ipc` if the server offers it and it works from your process. If the server offers it but it does not work, for example because the server runs on another machine, GPU arrays are copied through the host with a one-time warning saying why. A server that offers no GPU transport, such as one started by `Tesseract.from_source` or `Tesseract.from_image` without `gpu_transport`, gets host copies without a warning.
- Calls compiled for CPU always go through the host and never run the check.

To always copy GPU arrays through the host, pass a view of the Tesseract:

```python
apply_tesseract(tess.with_encoding(gpu_transport="none"), inputs)
```

## Dtypes

An array input whose dtype differs from its schema field, such as a float64 array for a `Float32` field under `jax_enable_x64`, is cast to the schema dtype on the device before the Tesseract is called. Both transports therefore send the Tesseract the same dtypes. JAX differentiates through the cast, so gradients come back in the dtype you passed.

- A float or complex field accepts any dtype that tesseract-core accepts for it. A value too large for the field's dtype becomes `inf`, as it would in any JAX cast.
- An integer or bool field only gets widening casts, such as int16 to `Int32`. Narrowing an integer or changing its sign, such as int64 to `Int32`, is left to tesseract-core. It checks that the values fit and casts on the host path, but raises under `cuda_ipc`. Pass the field's dtype yourself to use these with `cuda_ipc`, e.g. `x.astype(jnp.int32)`.
- With `jax_enable_x64` off, a field with a 64-bit dtype, such as `Float64`, cannot be matched in JAX. tesseract-core widens the input on the host path, but `cuda_ipc` raises.

A Tesseract whose differentiable inputs and outputs have different dtypes, such as float32 in and float64 out, pays a host round trip under `cuda_ipc` when its `jacobian` endpoint is used. `jax.jacfwd` expects the Jacobian in the output dtype and `jax.jacrev` in the input dtype, and the endpoint returns one dtype whichever mode asks. The other mode casts the whole Jacobian through host memory. Pass `materialize_jacobian=False` to `apply_tesseract` to batch the JVP or VJP endpoint instead, which avoids the cast.

## In-process Tesseracts

A Tesseract loaded with `Tesseract.from_tesseract_api` receives host (NumPy) arrays by default. Create it with `gpu_transport="cuda_ipc"` to hand its endpoints the device buffers as they are, as objects exposing `__cuda_array_interface__` that are only valid during the call:

```python
tess = Tesseract.from_tesseract_api("tesseract_api.py", gpu_transport="cuda_ipc")
```

Despite its name, `cuda_ipc` involves no inter-process communication here. Its endpoints may return host or device arrays either way.

The endpoints must not modify these input arrays in place. They are JAX's own buffers, which hold the caller's arrays and may be read again by later operations, so an in-place change such as `x -= x.mean()` silently changes the caller's values. Compute into a new array instead (`x = x - x.mean()`). A served Tesseract is not affected, since it receives its own copy of each input.
