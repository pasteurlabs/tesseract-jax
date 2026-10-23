# Debugging pipelines

When building pipelines that chain multiple Tesseracts, it can be difficult to understand what values flow between steps — especially when gradients are involved. Tesseract-JAX provides {func}`~tesseract_jax.sow` and {func}`~tesseract_jax.save_intermediates` to help you inspect intermediate values and their derivatives, and {func}`~tesseract_jax.check_gradients` to find the Tesseract that breaks a gradient.

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

## Sharp edge: cotangent names can be misleading

When using reverse-mode AD (`jax.grad`, `jax.vjp`), the cotangent captured at a `sow` point is the gradient flowing _backwards_ through that point. This means the cotangent at `"after_tess1"` is really the gradient arriving _before_ `tess1` in the backward pass — i.e., the gradient of the loss with respect to the _output_ of `tess1`, not its input.

In a two-step pipeline:

```python
def my_pipeline(inputs):
    res = apply_tesseract(tess1, inputs)
    res = sow(res, "after_tess1")   # between tess1 and tess2
    res = apply_tesseract(tess2, res)
    return res["output"].sum()

grads, intermediates = save_intermediates(jax.grad(my_pipeline))(inputs)
```

The forward pass flows left-to-right: `inputs → tess1 → [sow] → tess2 → loss`

But the backward pass flows right-to-left: `loss → tess2 → [sow] → tess1 → inputs`

So `intermediates["after_tess1"]["cotangent"]` contains the gradient _after_ backpropagating through `tess2` but _before_ backpropagating through `tess1`. A name like `"between_tess1_and_tess2"` is less ambiguous than `"after_tess1"`, since it describes the _location_ in the graph rather than a direction.

## Summary of captured keys

The keys present in each intermediate's dictionary depend on which JAX transformations are active:

| Transformation         | Keys captured                      |
| ---------------------- | ---------------------------------- |
| Plain call             | `primal`                           |
| `jax.grad` / `jax.vjp` | `primal`, `cotangent`              |
| `jax.jvp`              | `primal`, `tangent`                |
| `jax.jacobian`         | `primal`, `cotangent` (per-column) |
| `jax.jacfwd`           | `primal`, `tangent` (per-column)   |

## Finding the Tesseract that breaks a gradient

When the gradient of a whole pipeline is wrong, {func}`~tesseract_jax.check_gradients` finds where it goes wrong. It compares the gradient of your loss with a central finite difference along a random direction. While it computes that gradient, it records the inputs each Tesseract receives, and then runs the same check on each Tesseract alone at those inputs. Only the `apply`, `abstract_eval` and `vector_jacobian_product` endpoints are used.

```python
from tesseract_jax import apply_tesseract, check_gradients

def loss_fn(params):
    h = apply_tesseract(encoder, {"x": params["x"]})["y"]
    u = apply_tesseract(solver, {"x": jnp.sin(h)})["y"]
    return jnp.sum(jnp.cos(u))

report = check_gradients(loss_fn, {"x": jnp.array([0.3, 0.7, 1.1])})
print(report)
```

Here both Tesseracts square their input and the VJP of `solver` is 5% too large. Neither has a title in its `tesseract_config.yaml`, so each row is labelled by its call site and its differentiable inputs and outputs:

```
Gradient check along a random direction (seed=0, rtol=0.01, steps 1e-01 .. 1e-04)

                calls      grad.d   fin. diff  rel. error    step  result
  pipeline              -1.57e+00   -1.50e+00    4.76e-02   1e-03  FAIL

  #1 x -> y         1    3.22e+00    3.22e+00    1.38e-16   1e-01  ok
  #2 x -> y         1   -1.68e+00   -1.60e+00    4.76e-02   1e-01  FAIL

#2 x -> y disagrees with a finite difference of its own apply, at the inputs it received in the pipeline. Start there.
```

`report.passed` is `True` only if the pipeline and every call pass, so `assert report.passed` works in a test. Each row of `report.calls` also holds the Tesseract (`row.tesseract`) and the inputs it received (`row.inputs`).

### Reading the verdict

- **"... Start there."** names the first Tesseract call that fails on its own, because it disagrees with its finite difference or because it raised. If its finite difference is too uncertain to be sure, the verdict says its gradient may still be right.
- **"Every Tesseract agrees ... The problem is in the JAX code between the Tesseracts"** means each Tesseract is right at the inputs it received, along three random directions, but the pipeline is off by more than ten times the error of any Tesseract. Look for a `jax.lax.stop_gradient`, an operation whose derivative is zero or undefined (`round`, `argmax`, an integer cast, `sqrt` at 0), or a `custom_vjp`.
- **"... by less than ten times the error of ..."** means no Tesseract call fails, but one is close enough to `rtol` that it could explain the pipeline's error. Rerun with a smaller `rtol`.
- **"... Not every call was checked ..."** means no checked call fails, but a Tesseract ran more often than it was checked, so a call that was skipped may be the wrong one. Rerun with `max_calls_per_site=None`.
- **"The finite difference of the pipeline is too uncertain ..."** means the check cannot tell whether the gradient is wrong, as happens near a minimum of the loss, with a noisy loss, or in float32.
- **"The gradient of loss_fn could not be computed"** means `jax.grad(loss_fn)` raised; the pipeline row shows the exception. The Tesseracts are still checked: their calls are recorded again from plain evaluations of `loss_fn`, and an input counts as depending on `params` if it changes when `params` does.

A finite difference is taken at each step in `steps`, relative to the RMS of each input. A row passes when the gradient is within `rtol` of one of them after allowing for its error bound, and fails only when three neighbouring finite differences have settled and the gradient is farther than `rtol` from them. Otherwise the row is flagged as too uncertain.

### Limits

- By default only the first execution of each call site is checked, so for a Tesseract called in a loop (`scan`, `fori_loop`, `t(t(x))`) an error that shows only in later calls is missed; the verdict then says so instead of blaming the JAX code between the Tesseracts. Pass `max_calls_per_site=n` to check the first `n` executions, or `None` to check every one. The `calls` column shows how often each site ran, or `1 of 4` when only some executions were checked.
- In float32 a correct pipeline often cannot be confirmed, especially when the loss or an output carries a large constant. Use `jax.config.update("jax_enable_x64", True)`.
- Only reverse mode and first derivatives are checked, along one random direction per row, and along two more for each call when the pipeline fails and every call passes. A pass is evidence, not proof; rerun with another `seed`.
- Only floating-point leaves of `params` are perturbed. The check is not re-entrant, and Tesseract calls made by other threads while it runs are recorded too.
