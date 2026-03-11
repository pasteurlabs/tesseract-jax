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
       res = apply_tesseract(t, {"a": {"v": x}, "b": {"v": y}})
       return res["vector_add"]["result"].sum()

   vector_sum(x, y) # success!

   # You can also use it with JAX transformations like JIT and grad
   vector_sum_jit = jax.jit(vector_sum)
   vector_sum_jit(x, y)

   vector_sum_grad = jax.grad(vector_sum)
   vector_sum_grad(x, y)
   ```

```{tip}
Now you're ready to jump into our [examples](https://github.com/pasteurlabs/tesseract-jax/tree/main/examples) for ways to use Tesseract-JAX.
```

## Sharp edges

- **Arrays vs. array-like objects**: Tesseract-JAX is stricter than Tesseract Core in that all array inputs to Tesseracts must be JAX or NumPy arrays, not just any array-like (such as Python floats or lists). As a result, you may need to convert your inputs to JAX arrays before passing them to Tesseract-JAX, including scalar values.

  ```python
  from tesseract_core import Tesseract
  from tesseract_jax import apply_tesseract

  tess = Tesseract.from_image("vectoradd_jax")
  with Tesseract.from_image("vectoradd_jax") as tess:
      apply_tesseract(tess, {"a": {"v": [1.0]}, "b": {"v": [2.0]}})  # ❌ raises an error
      apply_tesseract(tess, {"a": {"v": jnp.array([1.0])}, "b": {"v": jnp.array([2.0])}})  # ✅ works
  ```
- **Additional required endpoints**: Tesseract-JAX requires the [`abstract_eval`](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/api/endpoints.html#abstract-eval) Tesseract endpoint to be defined when used in conjunction with automatic differentiation and JAX transformations. This is because JAX, in these cases, mandates abstract evaluation of all operations before they are executed. Additionally, many gradient transformations like `jax.grad` require [`vector_jacobian_product`](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/api/endpoints.html#vector-jacobian-product) to be defined.

```{tip}
When creating a new Tesseract based on a JAX function, use `tesseract init --recipe jax` to define all required endpoints automatically, including `abstract_eval` and `vector_jacobian_product`.
```

- **Non-differentiable outputs**: When an output is marked as non-differentiable in the Tesseract API, its behavior differs by differentiation mode.

  In **forward mode (JVP)**, tangents for non-differentiable outputs are `NaN`, which propagates to any downstream computation that depends on them.

  In **reverse mode (VJP)**, cotangents for non-differentiable outputs must be symbolic zeros — i.e., produced by JAX itself (not manually constructed arrays). Passing any concrete value (including `jnp.zeros_like(...)`) for a non-differentiable output's cotangent raises a `ValueError`. This is intentional: if you see this error, it likely means you forgot to mark an output as `Differentiable[...]` in the Tesseract schema.

  To handle non-differentiable outputs correctly, use one of these strategies:

  **Pop strategy** — exclude the non-differentiable output from the function return value:

  ```python
  def f(inputs):
      res = apply_tesseract(tess, inputs)
      res.pop("nondiff_res")
      return res

  primals, f_vjp = jax.vjp(f, inputs)
  cotangents = f_vjp(primals)
  ```

  **`has_aux` strategy** — return it as an auxiliary output outside the pytree:

  ```python
  def f(inputs):
      res = apply_tesseract(tess, inputs)
      return res["result"], res["nondiff_res"]  # (differentiable, aux)

  primals, f_vjp, nondiff_res = jax.vjp(f, inputs, has_aux=True)
  cotangents = f_vjp(primals)
  ```

  **`stop_gradient` strategy** — wrap the non-differentiable output so JAX produces a symbolic zero cotangent for it automatically:

  ```python
  def f(inputs):
      res = apply_tesseract(tess, inputs)
      res["nondiff_res"] = jax.lax.stop_gradient(res["nondiff_res"])
      return res

  primals, f_vjp = jax.vjp(f, inputs)
  cotangents = f_vjp(primals)  # symbolic zero for nondiff_res is produced automatically
  ```

  Note that the cotangent pytree structure must always match the function's output pytree structure. If you exclude non-differentiable outputs from the return value (e.g. via `pop` or `has_aux`), including them in the cotangent will raise a `ValueError`:

  ```
  ValueError: unexpected tree structure of argument to vjp function:
    got PyTreeDef({'nondiff_res': *, 'result': *}), but expected PyTreeDef({'result': *})
  ```

- **Non-differentiable inputs**: Requesting gradients with respect to an input that is marked as non-differentiable in the Tesseract API raises an error. Use the `argnums` parameter (or equivalent) to ensure gradient transformations only target differentiable inputs:

  ```python
  # "a" is differentiable, "b" is not
  def loss_fn(a, b):
      c = apply_tesseract(vectoradd_tess, {"a": a, "b": b})["c"]
      return jnp.sum(c**2)

  jax.grad(loss_fn, argnums=0)(a, b)  # ✅ gradient w.r.t. "a"
  jax.grad(loss_fn, argnums=1)(a, b)  # ❌ raises an error — "b" is non-differentiable
  ```
