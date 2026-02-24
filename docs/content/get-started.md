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
- **Additional required endpoints**: Tesseract-JAX requires the [`abstract_eval`](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/api/endpoints.html#abstract-eval) Tesseract endpoint to be defined for all operations. This is because JAX mandates abstract evaluation of all operations before they are executed. Additionally, many gradient transformations like `jax.grad` require [`vector_jacobian_product`](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/api/endpoints.html#vector-jacobian-product) to be defined.

```{tip}
When creating a new Tesseract based on a JAX function, use `tesseract init --recipe jax` to define all required endpoints automatically, including `abstract_eval` and `vector_jacobian_product`.
```

- **Non-differentiable outputs**: When an output is marked as non-differentiable in the Tesseract API, its behavior differs by differentiation mode.

  In **forward mode (JVP)**, tangents for non-differentiable outputs are `NaN`, which propagates to any downstream computation that depends on them.

  In **reverse mode (VJP)**, non-differentiable outputs are silently excluded from the cotangent sum. If you accidentally leave such an output in the return value and pass a cotangent that includes it, you will get a `ValueError` due to a pytree structure mismatch:

  ```
  ValueError: unexpected tree structure of argument to vjp function:
    got PyTreeDef({'O': *, 'result': *}), but expected PyTreeDef({'result': *})
  ```

  To compute a VJP correctly when the Tesseract has non-differentiable outputs, exclude them from the return value in a closure:

  ```python
  def f(inputs):
      res = apply_tesseract(tess, inputs)
      res.pop("O")  # exclude non-differentiable output
      return res

  primals, f_vjp = jax.vjp(f, inputs)
  cotangents = f_vjp(primals)
  ```

  If you need the *value* of a non-differentiable output as part of a loss, use `jax.lax.stop_gradient` to prevent gradients from flowing through it:

  ```python
  def loss_fn(inputs):
      results = apply_tesseract(tess, inputs)
      y = results["y"]
      O = jax.lax.stop_gradient(results["O"])
      return jnp.sum(y**2 + O.sum())
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
