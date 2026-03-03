# Debugging pipelines

When building pipelines that chain multiple Tesseracts, it can be difficult to understand what values flow between steps — especially when gradients are involved. Tesseract-JAX provides {func}`~tesseract_jax.sow` and {func}`~tesseract_jax.save_intermediates` to help you inspect intermediate values and their derivatives.

## Basic usage

Use {func}`~tesseract_jax.sow` to tag any intermediate value with a name, then wrap your function with {func}`~tesseract_jax.save_intermediates` to extract tagged values:

```python
from tesseract_jax import apply_tesseract, sow, save_intermediates

def my_pipeline(inputs):
    res = apply_tesseract(tess1, inputs)
    res = sow(res, "after_tess1")  # tag this intermediate
    res = apply_tesseract(tess2, res)
    return res["output"].sum()

result, intermediates = save_intermediates(my_pipeline)(inputs)

# intermediates["after_tess1"]["primal"] contains the forward-pass value
print(intermediates["after_tess1"]["primal"])
```

{func}`~tesseract_jax.sow` acts as a pure identity function — it returns its input unchanged and has no effect on the computation. Tagged values are only captured when the function is wrapped with {func}`~tesseract_jax.save_intermediates`.

## Capturing gradients

When {func}`~tesseract_jax.save_intermediates` wraps a function that involves a gradient transformation, it captures derivatives alongside primal values automatically.

### With `jax.grad` (cotangents)

```python
import jax

grad_fn = jax.grad(my_pipeline)
grads, intermediates = save_intermediates(grad_fn)(inputs)

# Forward-pass value at the tagged point
intermediates["after_tess1"]["primal"]

# Cotangent (gradient flowing back through this point)
intermediates["after_tess1"]["cotangent"]
```

### With `jax.jvp` (tangents)

```python
def forward_with_jvp(x):
    primals, tangents = jax.jvp(my_pipeline, (x,), (dx,))
    return primals

result, intermediates = save_intermediates(forward_with_jvp)(inputs)

# Forward-pass value
intermediates["after_tess1"]["primal"]

# Tangent (derivative propagated forward through this point)
intermediates["after_tess1"]["tangent"]
```

### With `jax.vjp`

```python
def forward_with_vjp(x):
    primals, f_vjp = jax.vjp(my_pipeline, x)
    grads = f_vjp(jnp.ones_like(primals))
    return grads[0]

result, intermediates = save_intermediates(forward_with_vjp)(inputs)
intermediates["after_tess1"]["primal"]
intermediates["after_tess1"]["cotangent"]
```

## Multiple tagged values

You can tag as many intermediates as you like — each must have a unique name:

```python
def my_pipeline(inputs):
    res1 = apply_tesseract(tess1, inputs)
    res1 = sow(res1, "after_tess1")

    res2 = apply_tesseract(tess2, res1)
    res2 = sow(res2, "after_tess2")

    return res2["output"].sum()

grads, intermediates = save_intermediates(jax.grad(my_pipeline))(inputs)

# Inspect each step independently
print(intermediates["after_tess1"]["primal"])
print(intermediates["after_tess1"]["cotangent"])
print(intermediates["after_tess2"]["primal"])
print(intermediates["after_tess2"]["cotangent"])
```

## Tags for grouping

Use the `tag` parameter to group intermediates and capture only a subset:

```python
def my_pipeline(inputs):
    res = apply_tesseract(tess1, inputs)
    res = sow(res, "step1", tag="debug")

    res = apply_tesseract(tess2, res)
    res = sow(res, "step2", tag="checkpoints")

    return res["output"].sum()

# Only capture "debug" tagged values
result, debug_ints = save_intermediates(my_pipeline, tag="debug")(inputs)
assert "step1" in debug_ints
assert "step2" not in debug_ints
```

## Compatibility with `jax.jit`

{func}`~tesseract_jax.sow` works inside `jax.jit`-compiled functions. {func}`~tesseract_jax.save_intermediates` traces into JIT boundaries and captures intermediates correctly:

```python
@jax.jit
def my_pipeline(inputs):
    res = apply_tesseract(tess1, inputs)
    res = sow(res, "after_tess1")
    return res["output"].sum()

# Works as expected — intermediates are captured from inside jit
result, intermediates = save_intermediates(my_pipeline)(inputs)
```

```{note}
{func}`~tesseract_jax.save_intermediates` should be the **outermost** transformation. It works by rewriting the function's JAX program trace, so it needs to wrap everything else.
```

## Summary of captured keys

The keys present in each intermediate's dictionary depend on which JAX transformations are active:

| Transformation       | Keys captured                    |
|-----------------------|----------------------------------|
| Plain call            | `primal`                         |
| `jax.grad` / `jax.vjp` | `primal`, `cotangent`          |
| `jax.jvp`            | `primal`, `tangent`              |
| `jax.jacobian`       | `primal`, `cotangent` (per-column) |
| `jax.jacfwd`       | `primal`, `tangent` (per-column)   |
