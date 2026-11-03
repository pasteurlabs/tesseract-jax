# Tracing a Tesseract directly

For an in-process, pure-JAX Tesseract, `apply_tesseract` can inline the Tesseract's own Python code into the jaxpr instead of dispatching it through a host callback. Call `to_traceable` once to get a Tesseract that does this, then use it exactly as you would the original:

```python
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract, to_traceable

tess = to_traceable(Tesseract.from_tesseract_api("tesseract_api.py"))
out = jax.jit(lambda x: apply_tesseract(tess, {"x": x}))(x)
```

Since there is no host callback, XLA sees straight through the Tesseract boundary: it can fuse the Tesseract's own operations with the surrounding computation, and there is no Python round trip for every call. The call also works with no server and no HTTP, since the real `apply` (or `jacobian_vector_product`, `vector_jacobian_product`, `jacobian`) function runs directly.

`to_traceable` requires an in-process Tesseract built with `Tesseract.from_tesseract_api(...)`; it raises `ValueError` for a served (`HTTPClient`) Tesseract, which has no importable Python function to trace.

```{warning}
Every endpoint a traced Tesseract actually calls must return a plain `dict`, or build its result via `model_construct`, instead of `OutputSchema(...)` or any other schema's validating constructor. The validating constructor calls `np.asarray` on every array field, which raises on a JAX tracer. This applies to *every* `apply`, `jacobian_vector_product`, `vector_jacobian_product`, and `jacobian` function you want to trace, not just one of them. If you see a `TracerArrayConversionError` from inside your Tesseract's own code, this is almost always why.
```

```python
# ❌ raises TracerArrayConversionError under to_traceable
def apply(inputs: InputSchema) -> OutputSchema:
    return OutputSchema(result=inputs.x * 2.0)

# ✅ works under to_traceable (and the callback path, unchanged)
def apply(inputs: InputSchema) -> OutputSchema:
    return {"result": inputs.x * 2.0}
```

## What validation still happens

A traced call validates shapes and dtypes against the same schema `abstract_eval` already uses, but not full pydantic validation of array values — the whole point is to run before any array has a concrete value. A few things follow from that:

- An input array whose dtype differs from its field's declared dtype is cast only when that is an AD-safe cast: any cast into a float/complex field, or a widening cast into an integer/bool field. A narrowing or sign-changing integer cast (e.g. `int64` into an `Int32` field) is left as-is, so a real mismatch surfaces as a clear error at trace time instead of silently computing on a wrapped value — pass the field's own dtype yourself in that case.
- An array field you omit, relying on its schema default, still gets that default's real value, as long as the field it is nested under (if any) is a plain class reference. A default nested behind an unusual type at the _model_ level (not the common `Array[...] | None` optional-leaf case, which already works) may not resolve; pass the field explicitly in that case.
- Output values are **not** re-validated at all — see the warning above.
