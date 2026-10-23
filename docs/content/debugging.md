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

When the gradient of a whole pipeline is wrong, {func}`~tesseract_jax.check_gradients` finds where it goes wrong. It compares the gradient of your loss with a central finite difference along a random direction. While it computes that gradient, it records the inputs each Tesseract receives, and then runs the same check on each Tesseract alone at those inputs.

Only the `apply`, `abstract_eval` and `vector_jacobian_product` endpoints are used, so Tesseracts that implement only a VJP can be checked.

```python
from tesseract_jax import apply_tesseract, check_gradients

def loss_fn(params):
    h = apply_tesseract(encoder, {"x": params["x"]})["y"]
    u = apply_tesseract(solver, {"x": jnp.sin(h)})["y"]
    return jnp.sum(jnp.cos(u))

report = check_gradients(loss_fn, {"x": jnp.array([0.3, 0.7, 1.1])})
print(report)
```

Here both Tesseracts square their input and the VJP of `solver` is 5% too large. Neither has a `name` in its `tesseract_config.yaml`, so each row is labelled by its call site and its differentiable inputs and outputs:

```
Gradient check along a random direction (seed=0, rtol=0.01, steps 1e-01 .. 1e-04)

                calls      grad.d   fin. diff  rel. error    step  result
  pipeline              -1.57e+00   -1.50e+00    4.76e-02   1e-03  FAIL

  #1 x -> y         1    3.22e+00    3.22e+00    1.38e-16   1e-01  ok
  #2 x -> y         1   -1.68e+00   -1.60e+00    4.76e-02   1e-01  FAIL

#2 x -> y disagrees with a finite difference of its own apply, at the inputs it received in the pipeline. Start there.
```

`report.passed` is `True` only if the pipeline and every call pass, so `assert report.passed` works in a test.

### Reading the result

Start with the last line, the verdict:

- **"... Start there."** names the first Tesseract call that fails on its own, because it disagrees with its finite difference or because it raised. Fix that one first; a wrong VJP upstream can make later checks look odd. A row that fails for sure is named before one whose finite difference is too uncertain to tell (see Step size below); for the latter, the verdict adds that its gradient may still be right.
- **"Every Tesseract agrees ... The problem is in the JAX code between the Tesseracts"** means each Tesseract is right at the inputs it received, along three random directions, but the pipeline is not, by more than ten times the error of any Tesseract. Look at the code that connects them, for example a `jax.lax.stop_gradient`, an operation whose derivative is zero or undefined (`round`, `argmax`, an integer cast, a comparison, `sqrt` at 0), or a `custom_vjp`.
- **"Not every call was checked"** means a Tesseract ran more often than it was checked (see below), so the pipeline failure may come from a call that was skipped. Rerun with `max_calls_per_site=None` before looking between the Tesseracts.
- **"... by less than ten times the error of ..."** means no Tesseract call fails, but one is close enough to `rtol` that its error, grown along the pipeline, could explain the pipeline's. Rerun with a smaller `rtol`.
- **"The finite difference of the pipeline is too uncertain ..."** means no Tesseract call failed, and the finite difference of the pipeline is too uncertain to tell whether its gradient is wrong. See Limits below.
- **"The gradient of loss_fn could not be computed"** means `jax.grad(loss_fn)` raised, and the pipeline row shows the exception. Common causes are a `while_loop`, which reverse mode does not support, a `loss_fn` that calls `jax.grad` through a Tesseract, and a loss that is not a real scalar. The Tesseracts are still checked: their calls are recorded again from plain evaluations of `loss_fn`, and an input counts as depending on `params` if it changes when `params` does.

The columns are:

- `calls`: how often the call site ran, or `1 of 6` when only some executions were checked.
- `grad.d`: the gradient contracted with the random direction `d`.
- `fin. diff`: the central difference along `d` that was kept (see Step size below).
- `rel. error`: `|grad.d - fin. diff| / max(|grad.d|, |fin. diff|)`. A row passes when it is at most `rtol`.
- `step`: the step that gave that finite difference.
- `result`: `ok`, `FAIL`, or `ERROR` with the exception that was raised. A NaN or inf in the gradient, or a finite difference too uncertain to tell, is flagged in parentheses.

A row is labelled with its call site and the Tesseract's `name`, or with its differentiable inputs and outputs (`x -> y`) when it has none. Each row of `report.calls` also holds the Tesseract (`row.tesseract`) and the inputs it received (`row.inputs`).

### Step size

The direction is scaled by the root mean square of each input, so the steps are relative: a step of `1e-3` moves an input of size 2e11 by about 2e8, and an input of size 1 by about 1e-3. The finite difference is taken at each step in `steps`, by default `(1e-1, 1e-2, 1e-3, 1e-4)`.

No single step suits every function. A coarse step has a curvature error, which shrinks with the step; a fine one has noise and rounding error, which grow as the step shrinks. So each finite difference but the finest gets an error bound: how far it moved from the next coarser step, divided by the step ratio minus one (the curvature error shrinks at least in proportion to the step), plus how far it moves to the next finer step (the noise there is at least as large), plus what rounding the function to its precision can do at that step. The coarsest step takes its curvature bound from the finer step instead. The finite difference with the smallest bound is shown.

A row passes when the gradient is within `rtol` of one of these finite differences after allowing for its bound. It fails only when three neighbouring finite differences have settled within `rtol` of each other and the gradient is farther than `rtol` from the middle one by more than their spread. Otherwise the finite difference is too uncertain to tell, and the row fails with that flag. Failing needs firmer evidence than passing because a failing row is named in the verdict. A large constant in float32, for example a loss of `1e8 + f`, rounds away every change the steps make, so its rows are too uncertain rather than wrong.

`rtol` defaults to `1e-2`. A looser tolerance such as `0.1` lets a gradient that is 5% wrong pass.

The finite difference at every step is stored in `row.finite_differences`, and the bound of the one shown in `row.fd_bound`. `steps` needs at least three steps.

### Calls inside `jit`, `scan` and `vmap`

`loss_fn` can be jitted, and can use loops, conditionals and `vmap`. A call site is one Tesseract called with the same input structure, shapes and static arguments.

| Construct                                                               | What is recorded                                                 |
| ----------------------------------------------------------------------- | ---------------------------------------------------------------- |
| `jax.jit`                                                               | The same as without `jit`, even for a function compiled earlier. |
| `scan`, `fori_loop`, `while_loop`                                       | One call site that runs once per iteration.                      |
| `cond`                                                                  | Only the branch that was taken.                                  |
| `vmap` with `vmap_method="sequential"`                                  | One call per batch element, with unbatched shapes.               |
| `vmap` with `"expand_dims"`, `"broadcast_all"` or `"auto_experimental"` | One call with batched shapes, checked as that batched call.      |
| `jax.checkpoint`                                                        | Like any other call. Its rerun in the backward pass is ignored.  |
| The same call twice, e.g. `t(t(x))`                                     | One call site that ran twice.                                    |
| Identical calls on identical values                                     | Possibly a single call, since XLA can merge them.                |
| Calls whose result the loss does not use                                | Nothing under `jit`, since JAX removes them.                     |

By default only the first execution of each call site is checked, since a Tesseract in a loop of 1000 iterations would otherwise be checked 1000 times. Pass `max_calls_per_site=n` to check the first `n`, or `None` to check every one.

### Tesseracts whose gradients are finite differences

Some Tesseracts compute their own VJP by finite differences, which the check cannot tell from the outside. For such a Tesseract a pass only shows its VJP is wired correctly. A failure usually points to its own step size at these inputs.

### Limits

A row fails only on firm evidence, so where finite differences are poor the check says it cannot tell rather than guess. It is still a finite-difference check, and some cases fool it:

- In float32 a correct pipeline often cannot be confirmed, especially when the loss or a Tesseract output carries a large constant, so `assert report.passed` can fail on a correct float32 pipeline. Run the check with `jax.config.update("jax_enable_x64", True)`.
- Noise in an output, for example from a solver tolerance, makes rows too uncertain. The verdict then names the first Tesseract with the note that it may be right, even when the error is between the Tesseracts.
- An error of about `rtol`, such as a VJP 1% off at the default `rtol`, is often reported as uncertain or passes. Use a smaller `rtol` to catch it.
- Near a minimum of the loss, for example at trained parameters, the gradient is small next to the curvature, so the pipeline row is often too uncertain. Check at another point.
- A kink or a feature finer than the step within reach of an input (`abs`, `clip`, an L1 or hinge loss with residuals near zero), or terms of the loss that cancel, can make the finite difference wrong, and then the verdict can name a correct Tesseract or blame correct code between them.
- For a Tesseract with several outputs, a wrong VJP for an output the loss does not depend on much can be missed, and the error then shows up as a pipeline failure blamed on the code between the Tesseracts.
- With the default `max_calls_per_site=1`, an error between the calls of a loop gets "Not every call was checked".
- Only reverse mode is checked, along random directions. The `jacobian_vector_product` and `jacobian` endpoints are not checked. An error in a few components of a large input can barely show along a direction: a VJP that gave a gradient to an input `apply` ignores was missed in about one run in ten. A pass is evidence, not proof; rerun with another `seed` to try other directions.
- Only first derivatives are checked. A `loss_fn` that itself calls `jax.grad` through a Tesseract is reported as an error. The loss must be a real scalar. Only floating-point leaves of `params` are perturbed; integer and boolean leaves are held fixed.
- The check is not re-entrant. While it runs, Tesseract calls made by other threads in the same process are recorded too.
- Each checked call costs about ten `apply` calls and one VJP, three times that when the pipeline fails and every call passes, and the pipeline about ten evaluations of `loss_fn`.

### Drilling into one Tesseract

To see which elements of a Tesseract's gradient are wrong, save the inputs it received and run `tesseract-runtime check-gradients` on them:

```python
import json
import numpy as np

row = report.calls[1]
inputs = jax.tree.map(lambda v: v.tolist() if isinstance(v, np.ndarray) else v, row.inputs)
with open("inputs.json", "w") as f:
    json.dump({"inputs": inputs}, f)
```

```bash
$ TESSERACT_API_PATH=solver/tesseract_api.py tesseract-runtime check-gradients @inputs.json --rtol 0.01
```

That tool perturbs one element at a time with an absolute step (`--eps`), and compares with a default `rtol` of 0.1, which lets the 5% error above pass, hence `--rtol 0.01`.
