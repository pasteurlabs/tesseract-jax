# Get started

## Quick start

```{note}
You need Python 3.12+ and [uv](https://docs.astral.sh/uv/getting-started/installation/), which Tesseract uses to build a separate virtual environment for each Tesseract. Docker is only needed to build container images.
```

1. Install Tesseract-JAX and get the example Tesseracts:

   ```bash
   $ pip install tesseract-jax
   $ git clone https://github.com/pasteurlabs/tesseract-jax
   ```

2. Use a Tesseract as part of a JAX program:

   ```python
   import jax
   import jax.numpy as jnp
   from tesseract_core import Tesseract
   from tesseract_jax import apply_tesseract

   # Serve the Tesseract in its own process (its environment is built on first use)
   t = Tesseract.from_source("tesseract-jax/examples/simple/vectoradd_jax/tesseract_api.py")
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

   `from_source` builds the environment from the Tesseract's requirements into `.tesseract-venv` next to `tesseract_api.py`, or runs on an existing interpreter passed as `python_executable=...`. Unlike a container, the Tesseract is not isolated from your filesystem and user.

3. To share or deploy the Tesseract, build the same folder into a container image (this step requires [Docker](https://docs.docker.com/engine/install/)) and replace `from_source` with `from_image`:

   ```bash
   $ tesseract build tesseract-jax/examples/simple/vectoradd_jax
   ```

   ```python
   t = Tesseract.from_image("vectoradd_jax")
   ```

   Use `Tesseract.from_url(...)` to reach a Tesseract that is already running elsewhere. `apply_tesseract` works the same way with all three.

```{seealso}
See [Batching strategies for jax.vmap](vmap-methods.md) for a guide on selecting the appropriate `vmap_method`, and the [Tesseract Core documentation](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/) for more on installing and serving Tesseracts.
```

```{tip}
Now you're ready to jump into our [examples](https://github.com/pasteurlabs/tesseract-jax/tree/main/examples) for ways to use Tesseract-JAX.
```

## Sharp edges

- **Additional required endpoints**: Tesseract-JAX requires the [`abstract_eval`](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/api/endpoints.html#abstract-eval) Tesseract endpoint to be defined to enable JAX tracing and FFI dispatch. To run a Tesseract that has no `abstract_eval` endpoint, call it directly through the Tesseract client instead. Additionally, many gradient transformations like `jax.grad` require [`vector_jacobian_product`](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/api/endpoints.html#vector-jacobian-product) to be defined.

```{tip}
When creating a new Tesseract based on a JAX function, use `tesseract init --recipe jax` to define all required endpoints automatically, including `abstract_eval` and `vector_jacobian_product`.
```

- **Non-differentiable inputs/outputs**: Differentiating through inputs or outputs not marked as `Differentiable[...]` in the Tesseract schema can raise a `ValueError` or produce `NaN` tangents. See the [Handling Differentiability](handling-differentiability.md) page for details and workarounds.

- **No JAX operations inside `from_tesseract_api` endpoints**: When using `Tesseract.from_tesseract_api(...)`, the `apply`, `vector_jacobian_product`, and `jacobian_vector_product` functions in your `tesseract_api.py` execute inside JAX FFI callbacks. **Using `jax.numpy` or any other JAX operation that allocates arrays in these functions can cause deadlocks**, because JAX's runtime is already holding a lock during the callback.

  Use plain NumPy instead, or serve the Tesseract in its own process with `Tesseract.from_source(...)` as in the quick start:

  ```python
  # ❌ Bad — will deadlock under jit/grad
  import jax.numpy as jnp

  def apply(inputs):
      return OutputSchema(c=jnp.sin(inputs.a))

  # ✅ Good — use numpy for in-process Tesseracts, or jnp via from_source
  import numpy as np

  def apply(inputs):
      return OutputSchema(c=np.sin(inputs.a))
  ```

  ```{note}
  This only affects `from_tesseract_api` (in-process execution). Tesseracts served in a separate process (`from_source`, `from_image`, or `from_url`) are not subject to this restriction.
  ```

- **Tesseracts are assumed pure functions of their inputs.** Tesseract-JAX lowers each
  endpoint call as a pure operation, which is what allows repeated identical calls to be
  collapsed into a single request. Where purity does not hold, the compiler is free to
  surprise you under `jax.jit`. Specifically:
  - **A call whose result is provably unused may not happen.**

    ```python
    @jax.jit
    def unused_result(a):
        _ = apply_tesseract(tess, inputs)["c"]  # not called: nothing depends on it
        return a * 2.0

    threshold_ok = False  # a concrete value, not a traced argument

    @jax.jit
    def dead_branch():
        # the predicate is known at compile time, so the branch is dead
        return jnp.where(threshold_ok, apply_tesseract(tess, inputs)["c"], 0.0)
    ```

    If the endpoint has an observable side effect (writing a file, logging to a tracking
    server), that side effect will not happen either, and neither will any error it
    would have raised. `abstract_eval` is still called while tracing, so a Tesseract
    that fails abstract validation still fails.

    This cuts both ways: guarding a call you know would be rejected is a legitimate way
    to avoid it, as long as the guard is something the compiler can evaluate. A guard on
    a traced value cannot be folded, so the call still happens.

  - **How many times a call happens is not guaranteed.** Two identical calls in one
    traced function may be collapsed into one, and the compiler is in principle free to
    recompute a call to save memory. An endpoint that returns different results for
    identical inputs, such as one sampling without a seed input or reading mutable
    external state, can therefore be called once where you expected twice, with both
    results being the same value.

  - **Ordering is not guaranteed** relative to other host callbacks such as
    `jax.debug.print`.

  If you have a Tesseract that genuinely depends on being called a particular number of
  times, or in a particular order, please
  [open an issue](https://github.com/pasteurlabs/tesseract-jax/issues) describing the
  workflow.
