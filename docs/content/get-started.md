# Get started

## Quick start

```{note}
Before proceeding, make sure you have a [working installation of Docker](https://docs.docker.com/engine/install/) and a modern Python installation (Python 3.10+).
```

```{seealso}
For more detailed installation instructions, please refer to the [Tesseract Core documentation](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/introduction/installation.html).
```

1. Install Tesseract-JAX:

   ```bash
   $ pip install tesseract-jax
   ```

2. Build an example Tesseract:

   ```bash
   $ git clone https://github.com/pasteurlabs/tesseract-jax
   $ tesseract build tesseract-jax/examples/simple/vectoradd_jax
   ```

3. Use it as part of a JAX program:

   ```python
   import jax
   import jax.numpy as jnp
   from tesseract_core import Tesseract
   from tesseract_jax import apply_tesseract

   # Load the Tesseract
   t = Tesseract.from_image("vectoradd_jax")
   t.serve()

   # Run it with JAX
   x = jnp.ones((1000,))
   y = jnp.ones((1000,))

   def vector_sum(x, y):
       res = apply_tesseract(t, {"a": {"v": x}, "b": {"v": y}}, vmap_method="sequential")
       return res["vector_add"]["result"].sum()

   vector_sum(x, y) # success!

   # You can also use it with JAX transformations like JIT and grad
   vector_sum_jit = jax.jit(vector_sum)
   vector_sum_jit(x, y)

   vector_sum_grad = jax.grad(vector_sum)
   vector_sum_grad(x, y)

   # vmap requires an explicit vmap_method — "sequential" is safe but slow
   # while "auto_experimental" or "expand_dims" is more efficient for Tesseracts that support batching.
   vector_sum_vmap = jax.vmap(vector_sum)
   vector_sum_vmap(x.reshape(10, 100), y.reshape(10, 100))
   ```

```{seealso}
See [Batching strategies for jax.vmap](vmap-methods.md) for a guide on selecting the appropriate `vmap_method`.
```

```{tip}
Now you're ready to jump into our [examples](https://github.com/pasteurlabs/tesseract-jax/tree/main/examples) for ways to use Tesseract-JAX.
```

## Sharp edges

- **Additional required endpoints**: Tesseract-JAX requires the [`abstract_eval`](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/api/endpoints.html#abstract-eval) Tesseract endpoint to be defined when used in conjunction with automatic differentiation and JAX transformations. This is because JAX, in these cases, mandates abstract evaluation of all operations before they are executed. Additionally, many gradient transformations like `jax.grad` require [`vector_jacobian_product`](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/api/endpoints.html#vector-jacobian-product) to be defined.

```{tip}
When creating a new Tesseract based on a JAX function, use `tesseract init --recipe jax` to define all required endpoints automatically, including `abstract_eval` and `vector_jacobian_product`.
```

- **Non-differentiable inputs/outputs**: Differentiating through inputs or outputs not marked as `Differentiable[...]` in the Tesseract schema can raise a `ValueError` or produce `NaN` tangents. See the [Handling Differentiability](handling-differentiability.md) page for details and workarounds.

- **No JAX operations inside `from_tesseract_api` endpoints**: When using `Tesseract.from_tesseract_api(...)`, the `apply`, `vector_jacobian_product`, and `jacobian_vector_product` functions in your `tesseract_api.py` execute inside JAX FFI callbacks. **Using `jax.numpy` or any other JAX operation that allocates arrays in these functions can cause deadlocks**, because JAX's runtime is already holding a lock during the callback.

  Use plain NumPy instead:

  ```python
  # ❌ Bad — will deadlock under jit/grad
  import jax.numpy as jnp

  def apply(inputs):
      return OutputSchema(c=jnp.sin(inputs.a))

  # ✅ Good — use numpy for in-process Tesseracts
  import numpy as np

  def apply(inputs):
      return OutputSchema(c=np.sin(inputs.a))
  ```

  ```{note}
  This only affects `from_tesseract_api` (in-process execution). Tesseracts served via Docker (`from_image`) run in a separate process and are not subject to this restriction.
  ```

- **Tesseracts are assumed pure functions** of their inputs. Tesseract-JAX lowers each
  endpoint call as a pure operation, which is what allows repeated identical calls to be
  collapsed into a single request. Where purity does not hold, the compiler is free to
  surprise you:
  - **A call whose result is provably unused may not happen.**

    ```python
    @jax.jit
    def unused_result(a):
        _ = apply_tesseract(tess, inputs)["c"]   # not called: nothing depends on it
        return a * 2.0

    threshold_ok = False   # a concrete value, not a traced argument

    @jax.jit
    def dead_branch():
        # the predicate is known at compile time, so the branch is dead
        return jnp.where(threshold_ok, apply_tesseract(tess, inputs)["c"], 0.0)
    ```

    If the endpoint has an observable side effect — writing a file, logging to a
    tracking server — that side effect will not happen either, and neither will any
    error it would have raised. `abstract_eval` is still called while tracing, so a
    Tesseract that fails abstract validation still fails.

    This cuts both ways: guarding a call you know would be rejected is a legitimate way
    to avoid it, as long as the guard is something the compiler can evaluate. A guard on
    a traced value cannot be folded, so the call still happens.

  - **How many times a call happens is not guaranteed.** Two identical calls in one
    traced function may be collapsed into one, and the compiler is in principle free to
    recompute a call to save memory. An endpoint that returns different results for
    identical inputs — sampling without a seed input, reading mutable external state —
    can therefore be called once where you expected twice, with both results being the
    same value.

  - **Ordering is not guaranteed** relative to other host callbacks such as
    `jax.debug.print`.

  If you have a Tesseract that genuinely depends on being called a particular number of
  times, or in a particular order, please
  [open an issue](https://github.com/pasteurlabs/tesseract-jax/issues) describing the
  workflow.
